from __future__ import annotations

import html
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

# Ensure project root is in sys.path when running script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_client import create_llm, safe_invoke
from agents.evidence_agent import _stable_source_key, _stable_evidence_id
from agents.verification_agent import ENTAILMENT, NEUTRAL, CONTRADICTION, UNVERIFIED


# ============================================================
# CITATION & VERIFICATION HELPERS
# ============================================================

def _normalize_doi(value):
    """Normalize DOI string by stripping URL prefixes, doi: schemes, and whitespace."""
    if not value:
        return ""
    val = str(value).strip().lower()
    val = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", val)
    return val.rstrip("/")


def _normalize_url(value):
    """Normalize URL by standardizing scheme, netloc, path, and ignoring query/fragments."""
    if not value:
        return ""
    val = str(value).strip()
    try:
        parts = urlsplit(val)
        if not parts.scheme and not parts.netloc:
            # Check if value is a DOI
            doi = _normalize_doi(val)
            if doi:
                return f"https://doi.org/{doi}"
            return val.rstrip("/").lower()
        return urlunsplit((
            parts.scheme.lower(),
            parts.netloc.lower(),
            parts.path.rstrip("/"),
            "",  # Ignore query parameters for deduplication.
            "",  # Ignore fragments for deduplication.
        ))
    except ValueError:
        return val.rstrip("/").lower()


def _source_aliases(source):
    """Identifiers that can connect equivalent source records."""
    aliases = []
    doi = _normalize_doi(source.get("doi"))
    url = _normalize_url(source.get("url"))
    # If DOI is absent but URL is a DOI resolver link, extract it
    if not doi and url:
        extracted_doi = _normalize_doi(url)
        if extracted_doi and extracted_doi != url:
            doi = extracted_doi
    if doi:
        aliases.append("doi:" + doi)
    if url:
        aliases.append("url:" + url)
    return aliases


def _verification_label_for_item(item, verification_results):
    """
    Resolve verification status using stable IDs first.
    Conflicting matching labels fail closed to UNVERIFIED.
    """
    results = verification_results or []
    evidence_id = item.get("evidence_id")
    source_key = item.get("source_key")
    claim = (item.get("claim") or "").strip()
    source_url = _normalize_url(item.get("source_url") or item.get("url"))

    # Strongest match: stable evidence identifier.
    if evidence_id:
        matches = [
            r for r in results
            if r.get("evidence_id") == evidence_id
        ]
        if matches:
            labels = {
                str(r.get("label") or "UNVERIFIED").upper()
                for r in matches
            }
            return labels.pop() if len(labels) == 1 else "UNVERIFIED"

    # Legacy-compatible match: source identity plus claim.
    if source_key and claim:
        matches = [
            r for r in results
            if r.get("source_key") == source_key
            and (r.get("claim") or "").strip() == claim
        ]
        if matches:
            labels = {
                str(r.get("label") or "UNVERIFIED").upper()
                for r in matches
            }
            return labels.pop() if len(labels) == 1 else "UNVERIFIED"

    # Final fallback for older records without stable identifiers.
    if source_url and claim:
        matches = [
            r for r in results
            if _normalize_url(r.get("source_url") or r.get("url")) == source_url
            and (r.get("claim") or "").strip() == claim
        ]
        if matches:
            labels = {
                str(r.get("label") or "UNVERIFIED").upper()
                for r in matches
            }
            return labels.pop() if len(labels) == 1 else "UNVERIFIED"

    # Claim-level fallback
    if claim:
        matches = [
            r for r in results
            if (r.get("claim") or "").strip() == claim
        ]
        if matches:
            labels = {
                str(r.get("label") or "UNVERIFIED").upper()
                for r in matches
            }
            return labels.pop() if len(labels) == 1 else "UNVERIFIED"

    return "UNVERIFIED"


def _audit_citations(report_body: str, source_count: int = 100) -> dict:
    """
    Audit inline citations against registered source count.
    Measures total markers, invalid markers, eligible claims, cited claims, and coverage.
    """
    # 1. Total and invalid marker counts
    all_markers = []
    for match in re.finditer(r"\[(\d+(?:\s*,\s*\d+)*)\]", report_body):
        parts = re.split(r"\s*,\s*", match.group(1))
        for p in parts:
            if p.isdigit():
                all_markers.append(int(p))

    total_marker_count = len(all_markers)
    valid_marker_count = sum(1 for m in all_markers if 1 <= m <= source_count)
    invalid_marker_count = sum(1 for m in all_markers if m < 1 or m > source_count)

    # 2. Candidate claims (using deterministic heuristic)
    lines = report_body.splitlines()
    in_code_block = False
    in_unverified_audit = False

    eligible_claims = 0
    cited_claims = 0

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue

        if "### Claim Verification Results" in stripped or "## Literature Summary Table" in stripped:
            in_unverified_audit = True
            continue
        if in_unverified_audit and stripped.startswith("## ") and not stripped.startswith("### Claim"):
            in_unverified_audit = False

        if in_unverified_audit:
            continue

        # Skip headers, table rows, horizontal rules, empty lines, and audit bullet lines
        if not stripped or stripped.startswith("#") or stripped.startswith("|") or stripped.startswith("---") or stripped.startswith("==="):
            continue
        if stripped.startswith("- ❌") or stripped.startswith("- ⚠️") or stripped.startswith("- ℹ️"):
            continue

        raw_sentences = re.split(r"(?<=[.!?])\s+", stripped)
        for s in raw_sentences:
            s_clean = s.strip()
            words = [w for w in s_clean.split() if w]
            if len(s_clean) < 25 or len(words) < 5:
                continue
            if s_clean.startswith("#") or s_clean.startswith("|"):
                continue

            s_markers = []
            for m in re.finditer(r"\[(\d+(?:\s*,\s*\d+)*)\]", s_clean):
                for p in re.split(r"\s*,\s*", m.group(1)):
                    if p.isdigit():
                        s_markers.append(int(p))

            has_registered = any(1 <= num <= source_count for num in s_markers)
            eligible_claims += 1
            if has_registered:
                cited_claims += 1

    if eligible_claims > 0:
        coverage_percent = round((cited_claims / eligible_claims) * 100, 2)
    else:
        coverage_percent = None

    return {
        "total_marker_count": total_marker_count,
        "valid_marker_count": valid_marker_count,
        "invalid_marker_count": invalid_marker_count,
        "eligible_claims": eligible_claims,
        "cited_claims": cited_claims,
        "eligible_sentences": eligible_claims,
        "cited_sentences": cited_claims,
        "coverage_percent": coverage_percent,
    }


def _clean(text, limit=None):
    text = html.unescape(html.unescape(str(text or ""))).replace("|", "/").replace("\n", " ").strip()
    if limit and len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0] + "…"
    return text


def _first_sentences(text, n=2, limit=300):
    text = re.sub(r"<[^>]+>", " ", html.unescape(str(text or "")))
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return _clean(" ".join(parts[:n]), limit)


def _format_authors(raw):
    if not raw:
        return "N/A"
    if isinstance(raw, str):
        return _clean(raw)
    names = [a.get("name", "") if isinstance(a, dict) else str(a) for a in raw]
    names = [n for n in names if n]
    if not names:
        return "N/A"
    return _clean(", ".join(names[:3]) + (" et al." if len(names) > 3 else ""))


# Compatibility aliases
_normalize_identifier = lambda s: (_source_aliases(s)[0] if _source_aliases(s) else "")

def _lookup_verification(item, lookup):
    results = lookup.get("results") or []
    label = _verification_label_for_item(item, results)
    ev_id = item.get("evidence_id")
    sk = item.get("source_key")
    claim = (item.get("claim") or "").strip()
    match = None
    if ev_id:
        match = next((r for r in results if r.get("evidence_id") == ev_id), None)
    if not match and sk and claim:
        match = next((r for r in results if r.get("source_key") == sk and (r.get("claim") or "").strip() == claim), None)
    if not match and claim:
        match = next((r for r in results if (r.get("claim") or "").strip() == claim), None)
    conf = match.get("confidence") if match else None
    return label, conf

validate_citations = lambda body, sources: {
    **_audit_citations(body, source_count=len(sources)),
    "eligible_factual_claims": _audit_citations(body, source_count=len(sources))["eligible_claims"],
    "cited_factual_claims": _audit_citations(body, source_count=len(sources))["cited_claims"],
    "uncited_factual_claims": _audit_citations(body, source_count=len(sources))["eligible_claims"] - _audit_citations(body, source_count=len(sources))["cited_claims"],
    "citation_coverage_pct": _audit_citations(body, source_count=len(sources))["coverage_percent"],
    "citation_coverage_display": f"{_audit_citations(body, source_count=len(sources))['coverage_percent']}%" if _audit_citations(body, source_count=len(sources))['coverage_percent'] is not None else "Unavailable (0 eligible claims)",
    "total_citation_markers": _audit_citations(body, source_count=len(sources))["total_marker_count"],
    "valid_citation_markers": _audit_citations(body, source_count=len(sources))["valid_marker_count"],
    "invalid_citation_markers": _audit_citations(body, source_count=len(sources))["invalid_marker_count"],
    "invalid_marker_numbers": [int(m) for m in re.findall(r"\[(\d+)\]", body) if int(m) > len(sources)],
    "uncited_examples": [],
}
_build_verification_lookup = lambda vr: {"results": vr}
format_citation_audit = lambda metrics: (
    "\n\n---\n\n"
    "## Citation Integrity Audit\n\n"
    f"- Eligible factual-claim candidates: {metrics.get('eligible_claims', metrics.get('eligible_factual_claims', 0))}\n"
    f"- Candidates with at least one registered citation: {metrics.get('cited_claims', metrics.get('cited_factual_claims', 0))}\n"
    f"- Claim-level citation coverage: {metrics.get('coverage_percent', metrics.get('citation_coverage_pct'))}%\n"
    f"- Invalid citation marker occurrences: {metrics.get('invalid_marker_count', metrics.get('invalid_citation_markers', 0))}\n"
    f"- Total inline citation marker occurrences: {metrics.get('total_marker_count', metrics.get('total_citation_markers', 0))}\n\n"
    "**Methodology note:** Coverage uses a deterministic sentence-length heuristic."
)


def build_citation_registry(literature_sources=None, web_sources=None, evidence=None):
    """Construct deduplicated citation registry and lookup dictionary."""
    citation_sources = []
    citation_numbers = {}
    alias_to_number = {}

    all_retrieved_sources = (literature_sources or []) + (web_sources or [])
    for src in all_retrieved_sources:
        url = (src.get("url") or "").strip()
        aliases = _source_aliases(src)
        if not aliases and not url:
            continue

        existing_number = next((alias_to_number[a] for a in aliases if a in alias_to_number), None)
        if existing_number is None and url and url in citation_numbers:
            existing_number = citation_numbers[url]

        if existing_number is None:
            citation_number = len(citation_sources) + 1
            citation_sources.append({
                "citation_number": citation_number,
                "title": src.get("title", "Untitled"),
                "url": url,
                "venue": src.get("journal_name", src.get("venue", "")),
                "issn": src.get("issn", ""),
                "doi": src.get("doi", ""),
                "is_retracted": src.get("is_retracted", str(src.get("retraction_status", "")).upper() == "RETRACTED"),
                "verified_indexes": src.get("verified_indexes", []),
                "confidence_level": src.get("identity_confidence", src.get("confidence_level", "")),
                "harvested_from": src.get("source") or ("Semantic Scholar Graph API" if ("10." in url or "semanticscholar.org" in url) else "Web Search"),
            })
        else:
            citation_number = existing_number

        if url:
            citation_numbers[url] = citation_number
        for a in aliases:
            alias_to_number[a] = citation_number

    for item in evidence or []:
        source_url = (item.get("source_url") or "").strip()
        if not source_url and not item.get("doi"):
            continue
        item_source = {"doi": item.get("doi", ""), "url": source_url}
        aliases = _source_aliases(item_source)

        citation_number = next((alias_to_number[a] for a in aliases if a in alias_to_number), None)
        if citation_number is None and source_url and source_url in citation_numbers:
            citation_number = citation_numbers[source_url]

        if citation_number is None:
            citation_number = len(citation_sources) + 1
            citation_sources.append({
                "citation_number": citation_number,
                "title": item.get("source_title", "Untitled"),
                "url": source_url,
                "venue": item.get("venue", ""),
                "issn": item.get("issn", ""),
                "doi": item.get("doi", ""),
                "is_retracted": item.get("is_retracted", str(item.get("retraction_status", "")).upper() == "RETRACTED"),
                "verified_indexes": item.get("verified_indexes", []),
                "confidence_level": item.get("confidence_level", ""),
                "harvested_from": item.get("harvested_from", "Evidence Extraction"),
            })

        if source_url:
            citation_numbers[source_url] = citation_number
        for a in aliases:
            alias_to_number[a] = citation_number

    combined_lookup = {**citation_numbers, **alias_to_number}
    return citation_sources, combined_lookup


# ============================================================
# LITERATURE TABLE BUILDER
# ============================================================

def build_literature_table(literature_sources, citation_sources, evidence,
                           unverified_pairs, gaps_text, profiles=None, verification_results=None):
    """Table: S.No. | Title | Author | Quality Indexing / Journal Name | Observed Claim, then Gaps."""
    cite_by_url = {c["url"]: c for c in citation_sources if c.get("url")}
    cite_by_doi = {_normalize_doi(c.get("doi")): c for c in citation_sources if c.get("doi")}

    profile_by_url = {(p.get("url") or "").strip().lower(): p for p in (profiles or [])}
    for p in (profiles or []):
        doi = _normalize_doi(p.get("doi"))
        if doi:
            profile_by_url[doi] = p
            profile_by_url[f"doi:{doi}"] = p

    supported_by_url = {}
    unverified_by_url = {}
    contradicted_by_url = {}
    not_checked_by_url = {}

    for item in evidence or []:
        url = (item.get("source_url") or item.get("url") or "").strip()
        claim = (item.get("claim") or "").strip()

        if not url or not claim:
            continue

        label = _verification_label_for_item(item, verification_results)

        if label == ENTAILMENT:
            bucket = supported_by_url
        elif label == NEUTRAL:
            bucket = unverified_by_url
        elif label == CONTRADICTION:
            bucket = contradicted_by_url
        else:
            bucket = not_checked_by_url

        for k in [url, url.lower().rstrip("/"), _normalize_url(url)]:
            claims_for_url = bucket.setdefault(k, [])
            if claim not in claims_for_url:
                claims_for_url.append(claim)

        doi = _normalize_doi(item.get("doi"))
        if doi:
            claims_for_doi = bucket.setdefault(f"doi:{doi}", [])
            if claim not in claims_for_doi:
                claims_for_doi.append(claim)

    rows = [
        "| S.No. | Title of the Paper | Author | Quality Indexing / Journal Name | Claims & Verification Status |",
        "|---|---|---|---|---|",
    ]
    n = 0
    for src in literature_sources or []:
        url = (src.get("url") or "").strip()
        doi = _normalize_doi(src.get("doi"))

        # Skip supplementary PDFs / attachments
        if url.endswith("/mm1") or (doi and doi.endswith("/mm1")) or src.get("title", "").lower().endswith(".pdf"):
            continue

        n += 1

        prof = (
            profile_by_url.get(url.lower())
            or (profile_by_url.get(doi) if doi else {})
            or (profile_by_url.get(f"doi:{doi}") if doi else {})
            or {}
        )
        meta = (
            cite_by_url.get(url)
            or (cite_by_doi.get(doi) if doi else {})
            or {}
        )

        raw_venue = (prof.get("journal_name") or prof.get("venue") or meta.get("venue") or src.get("journal_name") or src.get("venue") or "").strip()
        venue_note = ""
        if raw_venue.lower() in ("", "unknown", "none"):
            raw_venue = (prof.get("publisher") or "").strip()
            venue_note = " (publisher)" if raw_venue else ""
        raw_venue = raw_venue or "N/A"

        issn = prof.get("issn") or meta.get("issn") or src.get("issn")
        quality = raw_venue + venue_note
        if issn and issn != "N/A":
            quality += f" (ISSN {issn})"
        indexes = prof.get("indexed_in") or meta.get("verified_indexes") or []
        if indexes:
            quality += " — " + ", ".join(indexes)
        if prof.get("is_retracted") or meta.get("is_retracted"):
            quality += " ⚠️ RETRACTED"

        claim_lines = []
        verified_claims = (
            supported_by_url.get(url)
            or supported_by_url.get(url.lower().rstrip("/"))
            or supported_by_url.get(_normalize_url(url))
            or (supported_by_url.get(f"doi:{doi}") if doi else [])
            or []
        )
        for claim in verified_claims:
            claim_lines.append(
                f"• {_clean(claim, 220)} *(verified)*"
            )
            if len(claim_lines) >= 3:
                break

        cell = "<br>".join(claim_lines) if claim_lines else "No verified claims extracted"

        rows.append(
            f"| {n} | {_clean(src.get('title', 'Untitled'))} | {_format_authors(src.get('authors'))} "
            f"| {_clean(quality)} | {cell} |"
        )

    if n == 0:
        rows.append("| – | No academic papers found | – | – | – |")

    return "\n".join(rows) + "\n\n---\n\n**Gaps:**\n\n" + (gaps_text.strip() or "No gaps identified.")


# ============================================================
# MAIN REPORT AGENT ENTRY POINT
# ============================================================

def report_agent(
    topic,
    research_result,
    web_sources,
    literature_analysis,
    literature_sources,
    critic_result,
    evidence,
    verification_results=None,
    source_quality_table="",
    consensus_table="",
    source_quality_profiles=None,
):
    """Generate the final research report with verified source lists and citation audit."""

    llm = create_llm()

    # =========================================
    # UNIFIED CITATION NUMBERING ACROSS ALL SOURCES
    # =========================================
    citation_sources = []
    citation_numbers = {}       # Raw URL -> citation number
    alias_to_number = {}        # Normalized DOI/URL -> citation number

    all_retrieved_sources = (
        (literature_sources or []) + (web_sources or [])
    )
    for src in all_retrieved_sources:
        url = (src.get("url") or "").strip()
        aliases = _source_aliases(src)
        if not aliases and not url:
            continue

        # Reuse the number if any DOI/URL alias is already registered.
        existing_number = next(
            (alias_to_number[a] for a in aliases if a in alias_to_number),
            None,
        )
        if existing_number is None and url and url in citation_numbers:
            existing_number = citation_numbers[url]

        if existing_number is None:
            citation_number = len(citation_sources) + 1
            citation_sources.append({
                "title": src.get("title", "Untitled"),
                "url": url,
                "venue": src.get("journal_name", src.get("venue", "")),
                "issn": src.get("issn", ""),
                "doi": src.get("doi", ""),
                "is_retracted": src.get(
                    "is_retracted",
                    str(src.get("retraction_status", "")).upper() == "RETRACTED",
                ),
                "verified_indexes": src.get("verified_indexes", []),
                "confidence_level": src.get("identity_confidence", src.get("confidence_level", "")),
                "harvested_from": src.get("source") or ("Semantic Scholar Graph API" if ("10." in url or "semanticscholar.org" in url) else "Web Search"),
            })
        else:
            citation_number = existing_number

        if url:
            citation_numbers[url] = citation_number
        for a in aliases:
            alias_to_number[a] = citation_number

    evidence_sections = []
    for item in evidence or []:
        source_url = (item.get("source_url") or "").strip()
        if not source_url and not item.get("doi"):
            continue

        item_source = {
            "doi": item.get("doi", ""),
            "url": source_url,
        }
        aliases = _source_aliases(item_source)

        # Reuse a citation number assigned to this DOI or URL.
        citation_number = next(
            (alias_to_number[a] for a in aliases if a in alias_to_number),
            None,
        )
        if citation_number is None and source_url and source_url in citation_numbers:
            citation_number = citation_numbers[source_url]

        if citation_number is None:
            citation_number = len(citation_sources) + 1
            citation_sources.append({
                "title": item.get("source_title", "Untitled"),
                "url": source_url,
                "venue": item.get("venue", ""),
                "issn": item.get("issn", ""),
                "doi": item.get("doi", ""),
                "is_retracted": item.get(
                    "is_retracted",
                    str(item.get("retraction_status", "")).upper() == "RETRACTED",
                ),
                "verified_indexes": item.get("verified_indexes", []),
                "confidence_level": item.get("confidence_level", ""),
                "harvested_from": item.get("harvested_from", "Evidence Extraction"),
            })

        if source_url:
            citation_numbers[source_url] = citation_number
        for a in aliases:
            alias_to_number[a] = citation_number

        verif_status = _verification_label_for_item(item, verification_results)

        evidence_sections.append(
            f"CITATION [{citation_number}]\n"
            f"SOURCE_ID: {item.get('source_id', '')}\n"
            f"SOURCE_KEY: {item.get('source_key', '')}\n"
            f"EVIDENCE_ID: {item.get('evidence_id', '')}\n"
            f"TITLE: {item.get('source_title', '')}\n"
            f"URL: {source_url}\n"
            f"VERIFICATION_STATUS: {verif_status}\n\n"
            f"EVIDENCE:\n{item.get('evidence', '')}\n\n"
            f"CLAIM:\n{item.get('claim', '')}"
        )

    evidence_information = "\n\n".join(evidence_sections)
    if not evidence_information:
        evidence_information = "No extracted evidence was available."

    citation_registry_information = "\n".join(
        f"[{index}] {source.get('title', 'Untitled')}\n"
        f"URL: {source.get('url', '')}\n"
        f"DOI: {source.get('doi', '')}"
        for index, source in enumerate(citation_sources, start=1)
    )
    if not citation_registry_information:
        citation_registry_information = "No sources registered."

    # =========================================
    # UNVERIFIED CLAIMS (Verification Node Audit)
    # =========================================
    valid_unverified = []
    unverified_section = ""
    if verification_results:
        valid_unverified = [
            item
            for item in (verification_results or [])
            if str(item.get("label", UNVERIFIED)).upper() != ENTAILMENT
            and len((item.get("claim") or "").strip()) >= 5
        ]

        if valid_unverified:
            unverified_section = "\n\n### Claim Verification Results\n\n"
            unverified_section += (
                "The following claims were extracted from sources but were not confirmed as directly supported by our NLI verification model. "
                "This list includes unsupported, contradicted, and unchecked claims that should be treated with additional scrutiny:\n\n"
            )

            for uc in valid_unverified:
                label = str(uc.get("label", UNVERIFIED)).upper()
                if label == CONTRADICTION:
                    icon = "❌"
                elif label == NEUTRAL:
                    icon = "⚠️"
                else:
                    icon = "ℹ️"
                conf = f" (confidence: {uc['confidence']})" if uc.get("confidence") else ""

                uc_url = uc.get("source_url", "")
                aliases = _source_aliases({"doi": uc.get("doi", ""), "url": uc_url})
                cit_num = next((alias_to_number[a] for a in aliases if a in alias_to_number), None)
                if cit_num is None and uc_url in citation_numbers:
                    cit_num = citation_numbers[uc_url]

                cit_prefix = f"[{cit_num}] " if cit_num else ""

                unverified_section += f"- {icon} \"{uc.get('claim', '')}\"\n"
                unverified_section += f"  Source: {cit_prefix}[{uc.get('source_title', 'Untitled')}]({uc.get('source_url', '')}) | Status: {label}{conf}\n\n"

    if hasattr(critic_result, "decision"):
        critic_text = f"Decision: {critic_result.decision}\nReason: {critic_result.reason}\nKnowledge Gap: {getattr(critic_result, 'knowledge_gap', '')}"
    else:
        critic_text = str(critic_result)

    # =========================================
    # REPORT PROMPT
    # =========================================

    prompt = f"""
You are the final report-writing agent.

Research topic:
{topic}

========================================
GENERAL WEB RESEARCH
========================================

{research_result}

========================================
LITERATURE ANALYSIS
========================================

{literature_analysis}

========================================
CRITIC REPORT
========================================

{critic_text}

========================================
AUTHORITATIVE CITATION REGISTRY
========================================

{citation_registry_information}

This registry defines the valid citation-number-to-source mapping.
The registry is not itself evidence for a research finding.
Cite a source for a factual claim only when the supplied evidence
supports that claim. Never invent or renumber citation markers.

========================================
EVIDENCE AND SUPPORTED CLAIMS
========================================

{evidence_information}

========================================
TASK & STRICT SCOPE
========================================

Write ONLY the main body of the research report.
The scope is strictly '{topic}'. Do not narrow or drift the topic solely to a sub-aspect.
Name ONLY papers and models that appear in the provided EVIDENCE snippets. Do not cite papers outside the evidence list.
MANDATORY ENTITY RULE: Only mention organizations, companies, products, benchmarks, or models that explicitly appear in the provided EVIDENCE section. Do NOT mention outside organizations (such as OpenAI, Gemini, IBM, ChatGPT) unless they are verbatim in the evidence.

Use exactly these sections:

# Research Report

## 1. Executive Summary

## 2. General Research Findings

## 3. Literature Review

## 4. Critical Analysis

## 5. Knowledge Gaps

## 6. Recommendations

## 7. Conclusion

Do NOT create:
- Section 8
- Section 9
- a References section
- a bibliography
- a source list

Those sections will be created separately by Python.

========================================
SECTION RULES & EMPIRICAL RIGOR
========================================

CRITICAL REQUIREMENT: AVOID VAGUE GENERALIZATIONS AND DO NOT INVENT PLACEHOLDERS.
Do NOT write high-level fluff such as "AI is transforming the industry" or copy abstract templates.
Do NOT invent benchmark names, model names, or statistics if they are not in the evidence.
If the evidence does not contain specific percentages, describe the qualitative experimental findings accurately.

### 1. Executive Summary
Summarize:
- Research scope on {topic} and specific technical questions investigated
- Major empirical findings directly supported by the reviewed sources with citation tags [N]
- Primary limitations and conflicting findings identified across sources
- Evidence-grounded concluding assessment

### 2. General Research Findings
Discuss practical findings from the general web research.
- Focus on real-world implementations, industry reports, and empirical studies.
- Anchor all claims with their matching inline citation tags [N].

### 3. Literature Review
Deep-dive into the peer-reviewed academic literature provided:
- Compare specific paper methodologies and experimental findings from the evidence.
- Report concrete empirical observations directly drawn from the evidence.
- Highlight conflicting or nuanced results between papers where experimental scopes differ.
- For each paper discussed, note its limitations as reported in the evidence.

### 4. Critical Analysis
Evaluate the rigor and reliability of the gathered evidence:
- Synthesize cross-paper findings and highlight where papers agree or differ.
- Discuss source-quality, peer-review status, and experimental constraints.
- Identify methodologies that require further validation.

### 5. Knowledge Gaps
Identify specific, actionable technical questions that remain unanswered in current literature on {topic}:
- Avoid clichés like "Further research is needed."
- Detail precisely what datasets, evaluation protocols, or technical questions are missing.

### 6. Recommendations
Provide concrete, actionable research and engineering recommendations directly targeting the gaps.

### 7. Conclusion
Synthesize the final evidence-grounded verdict without introducing unverified claims.

========================================
CITATION & EVIDENCE RULES
========================================

- MANDATORY: Every factual claim or empirical statement MUST include its exact inline citation number tag [N] (e.g. [1] or [2]).
- Do NOT cite papers only by author name or year without the bracketed tag [N].
- Rely directly on the provided EVIDENCE snippets.
- Use the citation number shown beside each evidence item in CITATION [N].
- Place its marker directly after each supported claim (e.g. "Source finding [1].").
- Treat VERIFICATION_STATUS as a required constraint on how evidence is described.
- Only use ENTAILMENT evidence as affirmative support for a factual claim.
- NEUTRAL means the evidence does not establish support or contradiction.
- CONTRADICTION means the evidence conflicts with the claim; describe this
  as a conflicting finding, not as supporting evidence.
- UNVERIFIED evidence must not be presented as an established finding.
- Do not treat NLI entailment as proof that a scientific claim is true.
- Every [N] must refer to a citation number supplied in the evidence context.
- Never invent citation numbers, sources, DOIs, URLs, statistics, or findings.
- If the supplied evidence does not support a requested statement, explicitly
  qualify it or state that the available evidence is insufficient.
- Do not use a citation merely because its source discusses a similar topic;
  the supplied passage must be relevant to the particular statement.
- Return ONLY Sections 1 through 7.
"""

    # =========================================
    # WRITE REPORT VIA LLM
    # =========================================

    fallback_body = (
        "# Research Report\n\n"
        "## 1. Executive Summary\n\n"
        "The report body could not be generated due to an LLM "
        "error. Raw research materials are included below.\n\n"
        "## 2. General Research Findings\n\n"
        f"{research_result or 'No general research was collected.'}\n\n"
        "## 3. Literature Review\n\n"
        f"{literature_analysis or 'No literature analysis was generated.'}\n\n"
        "## 4. Critical Analysis\n\n"
        f"{critic_result or 'No critique was generated.'}\n\n"
        "## 5. Knowledge Gaps\n\n"
        "Not enough information was available.\n\n"
        "## 6. Recommendations\n\n"
        "Re-run the report generation step once the LLM service "
        "is available.\n\n"
        "## 7. Conclusion\n\n"
        "This report is incomplete due to a report-generation "
        "failure."
    )

    report_body = safe_invoke(
        llm,
        prompt,
        default=fallback_body
    )

    m = re.search(r"##\s*5\.\s*Knowledge Gaps\s*(.*?)(?=\n##\s*6\.|\Z)", report_body, re.S)
    gaps_text = m.group(1).strip() if m else getattr(critic_result, "knowledge_gap", "") or ""

    # =========================================
    # SPLICING UNVERIFIED CLAIMS BEFORE SECTION 6
    # =========================================
    if unverified_section:
        if "## 6. Recommendations" in report_body:
            parts = report_body.split("## 6. Recommendations")
            report_body = parts[0] + unverified_section + "\n\n## 6. Recommendations" + parts[1]
        else:
            report_body += unverified_section

    # =========================================
    # CITATION INTEGRITY AUDIT
    # =========================================
    citation_metrics = _audit_citations(
        report_body,
        source_count=len(citation_sources),
    )
    coverage = citation_metrics["coverage_percent"]
    coverage_text = (
        f"{coverage:.2f}%"
        if coverage is not None
        else "Unavailable (no eligible factual claims detected)"
    )
    citation_audit_section = (
        "\n\n---\n\n"
        "## Citation Integrity Audit\n\n"
        f"- Eligible factual-claim candidates: "
        f"{citation_metrics['eligible_claims']}\n"
        f"- Candidates with at least one registered citation: "
        f"{citation_metrics['cited_claims']}\n"
        f"- Claim-level citation coverage: {coverage_text}\n"
        f"- Invalid citation marker occurrences: "
        f"{citation_metrics['invalid_marker_count']}\n"
        f"- Total inline citation marker occurrences: "
        f"{citation_metrics['total_marker_count']}\n\n"
        "**Methodology note:** Coverage uses a deterministic sentence-length "
        "heuristic. It measures registered citation markers, not whether "
        "the cited passage semantically supports each claim. Invalid marker "
        "counts are based on citation numbers outside the registered range."
    )

    # =========================================
    # LITERATURE SUMMARY TABLE
    # =========================================
    unverified_pairs = {(uc.get("source_url", ""), uc.get("claim", "")) for uc in valid_unverified}
    literature_table = build_literature_table(
        literature_sources,
        citation_sources,
        evidence,
        unverified_pairs,
        gaps_text,
        profiles=source_quality_profiles,
        verification_results=verification_results,
    )
    literature_table_section = "\n\n---\n\n## Literature Summary Table\n\n" + literature_table

    # =========================================
    # SCIENTIFIC CONSENSUS MATRIX
    # =========================================
    consensus_section = ""
    if consensus_table:
        consensus_section = (
            "\n\n---\n\n"
            "## Cross-Paper Scientific Consensus Matrix\n\n"
            + consensus_table
        )

    # =========================================
    # SOURCE QUALITY & IDENTITY AUDIT TABLE
    # =========================================
    quality_audit_section = ""
    if source_quality_profiles:
        from agents.source_quality_agent import format_quality_table
        url_cit_map = {k.lower().strip(): v for k, v in citation_numbers.items() if isinstance(k, str)}
        for a, v in alias_to_number.items():
            if a.startswith("url:"):
                url_cit_map[a[4:].lower().strip()] = v
        source_quality_table = format_quality_table(source_quality_profiles, citation_map=url_cit_map)

    if source_quality_table:
        quality_audit_section = (
            "\n\n---\n\n"
            "## Source Quality & Verification Audit\n\n"
            + source_quality_table
        )

    # =========================================
    # APPEND PYTHON-VERIFIED CITATIONS (SOURCES)
    # =========================================
    if citation_sources:
        sources_lines = []
        for index, source in enumerate(citation_sources, start=1):
            title = html.unescape(source.get("title", "Untitled"))
            url = source.get("url", "")
            venue = html.unescape(source.get("venue", ""))
            issn = source.get("issn", "")
            harvested = source.get("harvested_from") or ("Semantic Scholar Graph API" if ("10." in url or "semanticscholar.org" in url) else "Web Search")
            verified_indexes = source.get("verified_indexes", [])
            is_retracted = source.get("is_retracted", False)

            line = f"[{index}] [{title}]({url})"
            meta_details = []

            meta_details.append(f"**Harvested From:** {harvested}")

            if venue:
                venue_str = f"**Venue:** *{venue}*"
                if issn and issn != "N/A":
                    venue_str += f" (ISSN: `{issn}`)"
                meta_details.append(venue_str)

            if verified_indexes:
                badges = " ".join(f"`[{idx}]`" for idx in verified_indexes)
                meta_details.append(f"**Verified Records:** {badges}")

            identity_conf = source.get("identity_confidence") or source.get("confidence_level", "")
            if identity_conf:
                meta_details.append(f"**Identity Verification:** `{identity_conf}`")

            if meta_details:
                line += "\n    " + " | ".join(meta_details)

            if is_retracted:
                line += "\n    ⚠️ **RETRACTION ALERT:** This paper has been officially flagged as RETRACTED by scholarly databases."

            sources_lines.append(line)

        sources_section = "\n\n".join(sources_lines)
    else:
        sources_section = "No cited sources were available."

    final_report = (
        report_body
        + literature_table_section
        + consensus_section
        + quality_audit_section
        + citation_audit_section
        + "\n\n"
        + "### Sources\n\n"
        + sources_section
    )

    return final_report