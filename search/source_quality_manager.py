"""
Source Quality Manager
======================
Verifies scholarly paper provenance, DOI authenticity, journal registration,
retraction status, and indexing signals via dual OpenAlex + Crossref verification.

Architecture:
                    Paper DOI
                       │
              ┌────────┴────────┐
              ↓                 ↓
          OpenAlex           Crossref
              │                 │
              └────────┬────────┘
                       ↓
               Compare metadata
                       ↓
              Identity Profile
"""

import difflib
import hashlib
import html
import json
import logging
import os
import re
from datetime import datetime, timezone
import requests

from config import (
    OPENALEX_BASE_URL,
    OPENALEX_MAILTO,
    OPENALEX_TIMEOUT_SECONDS,
    CROSSREF_VERIFY_BASE_URL,
    CROSSREF_VERIFY_TIMEOUT_SECONDS,
    CROSSREF_VERIFY_MAILTO,
    QUALITY_CACHE_DIR,
    QUALITY_CACHE_TTL_HOURS,
)

logger = logging.getLogger(__name__)


# ============================================================
# DISK CACHE (7-day TTL)
# ============================================================

def _cache_path(key_prefix: str, identifier: str) -> str:
    os.makedirs(QUALITY_CACHE_DIR, exist_ok=True)
    raw = f"{key_prefix}::{identifier.lower().strip()}"
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return os.path.join(QUALITY_CACHE_DIR, f"{h}.json")


def _get_cached_data(key_prefix: str, identifier: str):
    if not identifier:
        return None
    try:
        path = _cache_path(key_prefix, identifier)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            entry = json.load(f)
        cached_at = datetime.fromisoformat(entry.get("cached_at", "2000-01-01T00:00:00+00:00"))
        age_hours = (datetime.now(timezone.utc) - cached_at).total_seconds() / 3600
        if age_hours > QUALITY_CACHE_TTL_HOURS:
            os.remove(path)
            return None
        return entry.get("data")
    except Exception as err:
        logger.debug("[Quality Cache] Read error: %s", err)
        return None


def _set_cached_data(key_prefix: str, identifier: str, data: dict):
    if not identifier:
        return
    try:
        path = _cache_path(key_prefix, identifier)
        entry = {
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "data": data,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2)
    except Exception as err:
        logger.debug("[Quality Cache] Write error: %s", err)


# ============================================================
# NORMALIZATION & SIMILARITY HELPERS
# ============================================================

STOP_WORDS = {
    "a", "an", "the", "in", "on", "of", "and", "for", "with",
    "to", "at", "by", "from", "is", "are", "via", "using", "as"
}


def _clean_text(s: str) -> str:
    """Strip HTML tags, unescape entities, and clean punctuation/spaces."""
    if not s:
        return ""
    text = html.unescape(str(s))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return " ".join(text.split())


def _token_similarity(a: str, b: str) -> float:
    """
    Calculate combined token-overlap (Jaccard) and sequence similarity
    between two text strings (titles, venues, etc.).
    """
    clean_a = _clean_text(a)
    clean_b = _clean_text(b)
    if not clean_a or not clean_b:
        return 0.0
    if clean_a == clean_b:
        return 1.0

    words_a = [w for w in clean_a.split() if w not in STOP_WORDS] or clean_a.split()
    words_b = [w for w in clean_b.split() if w not in STOP_WORDS] or clean_b.split()

    tokens_a = set(words_a)
    tokens_b = set(words_b)

    if not tokens_a or not tokens_b:
        return 0.0

    jaccard = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
    seq_ratio = difflib.SequenceMatcher(None, clean_a, clean_b).ratio()

    return max(jaccard, seq_ratio)


def _journal_similarity(j1: str, j2: str) -> float:
    """Calculate journal/venue similarity supporting abbreviations and substrings."""
    if not j1 or not j2:
        return 0.0
    c1 = _clean_text(j1)
    c2 = _clean_text(j2)
    if not c1 or not c2:
        return 0.0
    if c1 == c2 or c1 in c2 or c2 in c1:
        return 1.0
    return _token_similarity(j1, j2)


def _author_overlap(authors_a: list, authors_b: list):
    """Return True for a match, False for a mismatch, None if unknown."""
    if not authors_a or not authors_b:
        return None

    def family_names(authors):
        names = set()
        for item in authors:
            if isinstance(item, str):
                if "," in item:
                    fam_part = item.split(",")[0].strip().lower()
                    if fam_part:
                        names.add(fam_part.split()[-1])
                else:
                    parts = item.strip().lower().split()
                    if parts:
                        names.add(parts[-1])
            elif isinstance(item, dict):
                family = item.get("family")
                if family:
                    names.add(str(family).strip().lower())
                else:
                    display = item.get("display_name") or item.get("name") or ""
                    if "," in display:
                        fam_part = display.split(",")[0].strip().lower()
                        if fam_part:
                            names.add(fam_part.split()[-1])
                    else:
                        parts = str(display).strip().lower().split()
                        if parts:
                            names.add(parts[-1])
        return names

    names_a = family_names(authors_a)
    names_b = family_names(authors_b)

    if not names_a or not names_b:
        return None

    # A first-author mismatch is a strong identity warning.
    if authors_a and authors_b:
        first_a = family_names([authors_a[0]])
        first_b = family_names([authors_b[0]])
        if first_a and first_b and not (first_a & first_b):
            return False

    return bool(names_a & names_b)


def _resolve_identity_status(profile: dict) -> str:
    """Separate identity-verification outcome from confidence level."""
    if profile.get("is_retracted"):
        return "RETRACTED"
    if profile.get("identity_conflict", False):
        return "IDENTITY_MISMATCH"
    confidence = profile.get("identity_confidence", "LOW")
    agreement = profile.get("metadata_agreement", "N/A")
    if confidence == "HIGH" and agreement == "HIGH":
        return "VERIFIED"
    if confidence == "MEDIUM":
        return "PROVISIONAL"
    return "UNVERIFIED"


# ============================================================
# API FETCHERS
# ============================================================

def _fetch_openalex(doi: str) -> tuple[str, dict]:
    """
    Fetch paper metadata from OpenAlex.
    Returns (status, data_dict).
    Status: 'VERIFIED', 'NOT_FOUND', 'TIMEOUT', 'RATE_LIMITED', 'UNAVAILABLE'.
    """
    url = f"{OPENALEX_BASE_URL}/works/https://doi.org/{doi}"
    headers = {"User-Agent": f"mailto:{OPENALEX_MAILTO}"}
    try:
        res = requests.get(url, headers=headers, timeout=OPENALEX_TIMEOUT_SECONDS)
        if res.status_code == 200:
            return "VERIFIED", res.json()
        elif res.status_code == 404:
            return "NOT_FOUND", {}
        elif res.status_code == 429:
            logger.warning("[Quality Manager] OpenAlex rate limited (HTTP 429)")
            return "RATE_LIMITED", {}
        else:
            logger.warning("[Quality Manager] OpenAlex returned HTTP %d for %s", res.status_code, doi)
            return "UNAVAILABLE", {}
    except requests.exceptions.Timeout:
        logger.warning("[Quality Manager] OpenAlex timeout for DOI: %s", doi)
        return "TIMEOUT", {}
    except Exception as err:
        logger.warning("[Quality Manager] OpenAlex error for %s: %s", doi, err)
        return "UNAVAILABLE", {}


def _fetch_crossref(doi: str) -> tuple[str, dict]:
    """
    Fetch paper metadata from Crossref.
    Returns (status, data_dict).
    Status: 'VERIFIED', 'NOT_FOUND', 'TIMEOUT', 'RATE_LIMITED', 'UNAVAILABLE'.
    """
    url = f"{CROSSREF_VERIFY_BASE_URL}/{doi}"
    headers = {"User-Agent": f"VeriResearchAgent/1.0 (mailto:{CROSSREF_VERIFY_MAILTO})"}
    try:
        res = requests.get(url, headers=headers, timeout=CROSSREF_VERIFY_TIMEOUT_SECONDS)
        if res.status_code == 200:
            msg = res.json().get("message", {})
            return "VERIFIED", msg
        elif res.status_code == 404:
            return "NOT_FOUND", {}
        elif res.status_code == 429:
            logger.warning("[Quality Manager] Crossref rate limited (HTTP 429)")
            return "RATE_LIMITED", {}
        else:
            logger.warning("[Quality Manager] Crossref returned HTTP %d for %s", res.status_code, doi)
            return "UNAVAILABLE", {}
    except requests.exceptions.Timeout:
        logger.warning("[Quality Manager] Crossref timeout (>= %ds) for DOI: %s", CROSSREF_VERIFY_TIMEOUT_SECONDS, doi)
        return "TIMEOUT", {}
    except Exception as err:
        logger.warning("[Quality Manager] Crossref error for %s: %s", doi, err)
        return "UNAVAILABLE", {}


# ============================================================
# METADATA PARSERS
# ============================================================

def _parse_openalex_work(data: dict) -> dict:
    """Normalize OpenAlex response into uniform schema."""
    primary_loc = data.get("primary_location") or {}
    source_obj = primary_loc.get("source") or {}

    authors = []
    for authorship in data.get("authorships") or []:
        author_info = authorship.get("author") or {}
        name = author_info.get("display_name") or ""
        if name:
            authors.append(name)

    issn_list = source_obj.get("issn") or []
    issn = issn_list[0] if issn_list else ""

    raw_indexes = [idx.lower() for idx in data.get("indexed_in", [])]
    indexes = ["OpenAlex (DOI record found)"]
    if "crossref" in raw_indexes:
        indexes.append("CrossRef (Indexed Work)")
    if "pubmed" in raw_indexes or data.get("ids", {}).get("pmid"):
        indexes.append("PubMed / MEDLINE")
    if source_obj.get("is_in_doaj"):
        indexes.append("DOAJ (Directory of Open Access Journals)")
    if "dblp" in raw_indexes:
        indexes.append("DBLP (Computer Science Bibliography)")

    return {
        "title": _clean_text(data.get("title", "")),
        "raw_title": data.get("title", ""),
        "journal_name": source_obj.get("display_name", "") or "",
        "issn": issn,
        "publisher": source_obj.get("host_organization_name", "") or "",
        "authors": authors,
        "year": data.get("publication_year"),
        "is_retracted": bool(data.get("is_retracted", False)),
        "is_doaj": bool(source_obj.get("is_in_doaj", False)),
        "citation_count": data.get("cited_by_count", 0),
        "indexed_in": indexes,
    }


def _parse_crossref_work(msg: dict) -> dict:
    """Normalize Crossref response into uniform schema."""
    titles = msg.get("title") or []
    raw_title = titles[0] if titles else ""
    cleaned_title = _clean_text(raw_title)

    containers = msg.get("container-title") or []
    raw_journal = containers[0] if containers else ""

    issns = msg.get("ISSN") or []
    issn = issns[0] if issns else ""

    publisher = msg.get("publisher", "") or ""

    authors = []
    for auth in msg.get("author") or []:
        fam = auth.get("family", "")
        given = auth.get("given", "")
        name = f"{given} {fam}".strip() if (given or fam) else auth.get("name", "")
        if name:
            authors.append(name)

    # Check for retraction in Crossref updates / updated-by / assertions / subtype
    is_retracted = False
    for upd_list in [msg.get("update-to") or [], msg.get("updated-by") or []]:
        for upd in upd_list:
            upd_type = (upd.get("type") or upd.get("label") or "").lower()
            if any(term in upd_type for term in ["retraction", "withdrawal", "retracted"]):
                is_retracted = True

    for assertion in msg.get("assertion") or []:
        ass_name = (assertion.get("name") or assertion.get("label") or "").lower()
        if any(term in ass_name for term in ["retraction", "retracted"]):
            is_retracted = True

    work_type = (msg.get("type") or "").lower()
    if "retraction" in work_type:
        is_retracted = True

    year = None
    issued = msg.get("issued") or {}
    date_parts = issued.get("date-parts") or [[]]
    if date_parts and date_parts[0]:
        year = date_parts[0][0]

    return {
        "title": cleaned_title,
        "raw_title": raw_title,
        "journal_name": raw_journal,
        "issn": issn,
        "publisher": publisher,
        "authors": authors,
        "year": year,
        "is_retracted": is_retracted,
        "indexed_in": ["CrossRef (Official DOI Registrar)"],
    }


# ============================================================
# MAIN VERIFICATION ENGINE
# ============================================================

def verify_single_paper(paper: dict) -> dict:
    """
    Verify a single paper's identity, retraction status, and journal registration
    using dual OpenAlex + Crossref verification.

    Parameters
    ----------
    paper : dict
        Contains 'title', 'doi', 'url', 'authors', 'year', 'venue', etc.

    Returns
    -------
    dict
        Structured Source Quality Profile with identity confidence and reasons.
    """
    raw_doi = (paper.get("doi") or "").strip()
    title = (paper.get("title") or "").strip()
    url = (paper.get("url") or "").strip()
    paper_venue = (paper.get("venue") or "").strip()
    paper_authors = paper.get("authors") or []

    # Clean DOI
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", raw_doi).strip()

    # Check full profile cache (keyed by DOI and cleaned title)
    paper_cache_key = f"{doi}::{_clean_text(title)}" if doi else _clean_text(title)
    cached_profile = _get_cached_data("profile", paper_cache_key)
    if cached_profile is not None:
        logger.debug("[Quality] Profile Cache HIT for: %s", paper_cache_key[:40])
        # Re-evaluate strong_identity exemption on cached profiles so updated rules take effect
        doi_in_both = (cached_profile.get("openalex_status") == "VERIFIED" and cached_profile.get("crossref_status") == "VERIFIED")
        if (
            cached_profile.get("title_similarity", 0.0) >= 0.80
            and cached_profile.get("authors_match") is True
            and doi_in_both
        ):
            cached_profile["identity_conflict"] = False
            if cached_profile.get("metadata_agreement") == "MISMATCH":
                cached_profile["metadata_agreement"] = "MODERATE"
            if cached_profile.get("identity_confidence") == "LOW":
                cached_profile["identity_confidence"] = "MEDIUM"
                cached_profile["confidence_level"] = "MEDIUM"

        cached_profile["identity_status"] = _resolve_identity_status(cached_profile)
        return cached_profile

    # Base profile structure
    profile = {
        "title": title,
        "doi": doi,
        "url": url,
        "doi_verified": False,
        "openalex_status": "SKIPPED",
        "crossref_status": "SKIPPED",
        "metadata_agreement": "N/A",
        "title_similarity": 0.0,
        "title_matched": False,
        "is_retracted": False,
        "retraction_details": "",
        "journal_name": paper_venue,
        "issn": "",
        "publisher": "",
        "is_doaj": False,
        "peer_reviewed": "Unknown / Unverified",
        "citation_count": 0,
        "indexed_in": [],
        "scopus_status": "Not independently verified by our system",
        "wos_status": "Not independently verified by our system",
        "identity_confidence": "LOW",
        "confidence_level": "LOW",
        "confidence_reason": "No DOI or metadata record found.",
        "identity_status": "UNVERIFIED",
        "identity_conflict": False,
        "authors_match": None,
        "journal_match": None,
    }

    # Case 2: Handle arXiv preprints directly (Skip Crossref)
    if "arxiv.org" in url.lower() or "arxiv" in doi.lower() or "arxiv" in paper_venue.lower():
        profile.update({
            "doi_verified": bool(doi),
            "openalex_status": "SKIPPED (Preprint)",
            "crossref_status": "SKIPPED (Preprint)",
            "metadata_agreement": "N/A (Preprint)",
            "title_matched": bool(title),
            "journal_name": "arXiv",
            "issn": "N/A (Preprint)",
            "peer_reviewed": "No (Preprint)",
            "indexed_in": ["arXiv (Preprint - Non-Peer-Reviewed)"],
            "identity_confidence": "MEDIUM",
            "confidence_level": "MEDIUM",
            "confidence_reason": "Authentic preprint, but not formally peer-reviewed.",
        })
        profile["identity_status"] = _resolve_identity_status(profile)
        _set_cached_data("profile", paper_cache_key, profile)
        return profile

    # Case 4a: Missing DOI
    if not doi:
        profile.update({
            "confidence_reason": "Paper lacks a standard DOI identifier.",
            "identity_confidence": "LOW",
            "confidence_level": "LOW",
        })
        profile["identity_status"] = _resolve_identity_status(profile)
        _set_cached_data("profile", paper_cache_key, profile)
        return profile

    # Retrieve or fetch external registry data for this DOI
    cached_reg = _get_cached_data("registry", doi)
    if cached_reg:
        oa_status = cached_reg.get("oa_status")
        oa_raw = cached_reg.get("oa_raw") or {}
        cr_status = cached_reg.get("cr_status")
        cr_raw = cached_reg.get("cr_raw") or {}
    else:
        logger.info("[Quality Manager] Querying OpenAlex & Crossref for DOI: %s", doi)
        oa_status, oa_raw = _fetch_openalex(doi)
        cr_status, cr_raw = _fetch_crossref(doi)
        _set_cached_data("registry", doi, {
            "oa_status": oa_status,
            "oa_raw": oa_raw,
            "cr_status": cr_status,
            "cr_raw": cr_raw,
        })

    profile["openalex_status"] = oa_status
    profile["crossref_status"] = cr_status

    oa_meta = _parse_openalex_work(oa_raw) if oa_status == "VERIFIED" else None
    cr_meta = _parse_crossref_work(cr_raw) if cr_status == "VERIFIED" else None

    # DOI Verified if found in either OpenAlex or Crossref
    profile["doi_verified"] = (oa_status == "VERIFIED" or cr_status == "VERIFIED")

    # ========================================================
    # 1. RETRACTION CHECK (OpenAlex takes strict precedence)
    # ========================================================
    is_retracted = False
    retraction_reasons = []

    if oa_meta and oa_meta.get("is_retracted"):
        is_retracted = True
        retraction_reasons.append("OpenAlex / Retraction Watch flagged paper as RETRACTED")
    if cr_meta and cr_meta.get("is_retracted"):
        is_retracted = True
        retraction_reasons.append("Crossref record flags publication as RETRACTED")

    if is_retracted:
        profile.update({
            "is_retracted": True,
            "retraction_details": "; ".join(retraction_reasons),
            "identity_confidence": "RETRACTED_WARNING",
            "confidence_level": "RETRACTED_WARNING",
            "confidence_reason": "CRITICAL: This paper has been officially retracted.",
            "journal_name": (oa_meta or cr_meta or {}).get("journal_name", paper_venue),
            "issn": (oa_meta or cr_meta or {}).get("issn", ""),
        })
        profile["identity_status"] = _resolve_identity_status(profile)
        _set_cached_data("profile", paper_cache_key, profile)
        return profile

    # ========================================================
    # 2. METADATA EXTRACTION & POPULATION
    # ========================================================
    resolved_journal = ""
    resolved_issn = ""
    resolved_publisher = ""
    resolved_citations = 0
    resolved_doaj = False
    resolved_indexes = []

    if oa_meta:
        resolved_journal = oa_meta["journal_name"] or resolved_journal
        resolved_issn = oa_meta["issn"] or resolved_issn
        resolved_publisher = oa_meta["publisher"] or resolved_publisher
        resolved_citations = oa_meta["citation_count"]
        resolved_doaj = oa_meta["is_doaj"]
        resolved_indexes.extend(oa_meta["indexed_in"])

    if cr_meta:
        if not resolved_journal:
            resolved_journal = cr_meta["journal_name"]
        if not resolved_issn:
            resolved_issn = cr_meta["issn"]
        if not resolved_publisher:
            resolved_publisher = cr_meta["publisher"]
        for idx in cr_meta["indexed_in"]:
            if idx not in resolved_indexes:
                resolved_indexes.append(idx)

    profile["journal_name"] = resolved_journal or paper_venue or "Unknown"
    profile["issn"] = resolved_issn
    profile["publisher"] = resolved_publisher
    profile["citation_count"] = resolved_citations
    profile["is_doaj"] = resolved_doaj
    profile["indexed_in"] = resolved_indexes

    if resolved_doaj:
        profile["peer_reviewed"] = "Yes (DOAJ Verified Peer Review)"
    else:
        profile["peer_reviewed"] = "Unknown / Unverified"

    # ========================================================
    # 3. METADATA COMPARISON & SIMILARITY AUDIT
    # ========================================================
    sim_paper_oa = _token_similarity(title, oa_meta["raw_title"]) if oa_meta else 0.0
    sim_paper_cr = _token_similarity(title, cr_meta["raw_title"]) if cr_meta else 0.0
    sim_oa_cr = _token_similarity(oa_meta["raw_title"], cr_meta["raw_title"]) if (oa_meta and cr_meta) else 0.0

    best_paper_sim = max(sim_paper_oa, sim_paper_cr)
    profile["title_similarity"] = round(best_paper_sim, 3)
    profile["title_matched"] = best_paper_sim >= 0.75

    # Authors overlap
    authors_match = None
    if oa_meta and cr_meta:
        authors_match = _author_overlap(oa_meta["authors"], cr_meta["authors"])

    if paper_authors:
        if oa_meta:
            oa_overlap = _author_overlap(paper_authors, oa_meta["authors"])
            if oa_overlap is False:
                authors_match = False
            elif oa_overlap is True and authors_match is not False:
                authors_match = True
        if cr_meta:
            cr_overlap = _author_overlap(paper_authors, cr_meta["authors"])
            if cr_overlap is False:
                authors_match = False
            elif cr_overlap is True and authors_match is not False:
                authors_match = True
    else:
        # Missing author metadata for the input paper cannot qualify for positive match
        if authors_match is not False:
            authors_match = None

    # Journal / Venue match
    journal_match = None
    # Treat venues containing 'ebook' as platform names to skip false-positive venue comparisons
    is_ebook_platform = any(
        "ebook" in str(v).lower()
        for v in [
            (oa_meta or {}).get("journal_name", ""),
            (cr_meta or {}).get("journal_name", ""),
            paper_venue
        ]
    )

    if (
        oa_meta
        and cr_meta
        and oa_meta.get("journal_name")
        and cr_meta.get("journal_name")
        and paper_venue
        and paper_venue.lower() not in ("unknown", "none", "")
        and not is_ebook_platform
    ):
        j_sim = _journal_similarity(
            oa_meta["journal_name"],
            cr_meta["journal_name"],
        )
        journal_match = j_sim >= 0.55

    doi_verified_in_both = (oa_status == "VERIFIED" and cr_status == "VERIFIED")
    strong_identity = (
        best_paper_sim >= 0.80
        and authors_match is True
        and doi_verified_in_both
    )

    profile["authors_match"] = authors_match
    profile["journal_match"] = journal_match
    profile["identity_conflict"] = (
        authors_match is False
        or ((journal_match is False) and not strong_identity)
        or (
            (oa_meta is not None or cr_meta is not None)
            and best_paper_sim < 0.70
        )
    )

    # ========================================================
    # 4. IDENTITY CONFIDENCE SYNTHESIS
    # ========================================================

    # SCENARIO A: Both OpenAlex and Crossref verified the DOI
    if oa_status == "VERIFIED" and cr_status == "VERIFIED":
        dual_registry_agrees = (sim_oa_cr >= 0.85)
        paper_matches_registry = (best_paper_sim >= 0.80)
        moderate_paper_match = (best_paper_sim >= 0.70)

        # HIGH requires positive evidence for all three checks.
        if (
            dual_registry_agrees
            and paper_matches_registry
            and authors_match is True
            and journal_match is True
        ):
            profile["metadata_agreement"] = "HIGH"
            profile["identity_confidence"] = "HIGH"
            profile["confidence_level"] = "HIGH"
            profile["confidence_reason"] = (
                f"Dual verification confirmed (OpenAlex + Crossref). "
                f"Metadata substantially agrees (Title sim: {sim_oa_cr:.2f}, Journal confirmed)."
            )
        elif (
            moderate_paper_match
            and (dual_registry_agrees or sim_oa_cr >= 0.60)
            and authors_match is not False
            and (journal_match is not False or strong_identity)
        ):
            profile["metadata_agreement"] = "MODERATE"
            profile["identity_confidence"] = "MEDIUM"
            profile["confidence_level"] = "MEDIUM"
            profile["confidence_reason"] = (
                f"DOI registered in both OpenAlex and Crossref, but metadata exhibits moderate differences "
                f"(Title sim: {best_paper_sim:.2f})."
            )
        else:
            profile["metadata_agreement"] = "MISMATCH"
            profile["identity_confidence"] = "LOW"
            profile["confidence_level"] = "LOW"
            profile["confidence_reason"] = (
                f"DOI exists, but registered metadata substantially conflicts with input paper details "
                f"(Title sim: {best_paper_sim:.2f})."
            )

    # SCENARIO B: OpenAlex verified, Crossref unavailable / timeout / rate limited
    elif oa_status == "VERIFIED" and cr_status in ("TIMEOUT", "UNAVAILABLE", "RATE_LIMITED"):
        profile["metadata_agreement"] = "SINGLE_SOURCE (Crossref unavailable)"
        if sim_paper_oa >= 0.70 and authors_match is not False and journal_match is not False:
            profile["identity_confidence"] = "MEDIUM"
            profile["confidence_level"] = "MEDIUM"
            profile["confidence_reason"] = (
                f"DOI verified via OpenAlex, but Crossref secondary verification was {cr_status.lower()}."
            )
        else:
            profile["identity_confidence"] = "LOW"
            profile["confidence_level"] = "LOW"
            profile["confidence_reason"] = (
                f"DOI found in OpenAlex, but title similarity low ({sim_paper_oa:.2f}) or metadata conflicts, and Crossref was {cr_status.lower()}."
            )

    # SCENARIO C: Crossref verified, OpenAlex unavailable / timeout / rate limited
    elif cr_status == "VERIFIED" and oa_status in ("TIMEOUT", "UNAVAILABLE", "RATE_LIMITED"):
        profile["metadata_agreement"] = "SINGLE_SOURCE (OpenAlex unavailable)"
        if sim_paper_cr >= 0.70 and authors_match is not False and journal_match is not False:
            profile["identity_confidence"] = "MEDIUM"
            profile["confidence_level"] = "MEDIUM"
            profile["confidence_reason"] = (
                f"DOI verified via Crossref, but OpenAlex primary verification was {oa_status.lower()}."
            )
        else:
            profile["identity_confidence"] = "LOW"
            profile["confidence_level"] = "LOW"
            profile["confidence_reason"] = (
                f"DOI found in Crossref, but title similarity low ({sim_paper_cr:.2f}) or metadata conflicts, and OpenAlex was {oa_status.lower()}."
            )

    # SCENARIO D: One verified, other NOT_FOUND
    elif (oa_status == "VERIFIED" and cr_status == "NOT_FOUND") or (cr_status == "VERIFIED" and oa_status == "NOT_FOUND"):
        profile["metadata_agreement"] = "SINGLE_SOURCE (Registry discrepancy)"
        if best_paper_sim >= 0.70 and authors_match is not False and journal_match is not False:
            profile["identity_confidence"] = "MEDIUM"
            profile["confidence_level"] = "MEDIUM"
            profile["confidence_reason"] = "DOI confirmed in one scholarly registry, but record absent in the second."
        else:
            profile["identity_confidence"] = "LOW"
            profile["confidence_level"] = "LOW"
            profile["confidence_reason"] = "DOI found in only one registry with weak title agreement or metadata mismatch."

    # SCENARIO E: Neither provider found the DOI (Both 404)
    elif oa_status == "NOT_FOUND" and cr_status == "NOT_FOUND":
        profile["metadata_agreement"] = "NOT_FOUND"
        profile["identity_confidence"] = "LOW"
        profile["confidence_level"] = "LOW"
        profile["confidence_reason"] = "DOI record was not found in either OpenAlex or Crossref."

    # SCENARIO F: Both services experienced timeout or error
    else:
        profile["metadata_agreement"] = "UNAVAILABLE"
        profile["identity_confidence"] = "LOW"
        profile["confidence_level"] = "LOW"
        profile["confidence_reason"] = (
            f"Verification services unreachable (OpenAlex: {oa_status}, Crossref: {cr_status})."
        )

    profile["identity_status"] = _resolve_identity_status(profile)
    _set_cached_data("profile", paper_cache_key, profile)
    return profile


def batch_verify_papers(papers: list) -> list:
    """Verify an entire list of academic papers."""
    results = []
    for paper in papers:
        profile = verify_single_paper(paper)
        results.append(profile)
    return results
