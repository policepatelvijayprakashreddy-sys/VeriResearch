"""
Source Quality Agent
====================
Orchestrates paper-level identity and journal quality audits.
Produces structured Source Quality Profiles for academic literature.

Inputs:
  literature_sources: list of academic source dicts from literature_agent.

Outputs:
  source_quality_profiles: list of verified paper profiles.
  quality_summary: aggregate statistics and markdown scorecard.
  retraction_warnings: any retracted papers that must be flagged or discarded.
"""

import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from search.source_quality_manager import batch_verify_papers

logger = logging.getLogger(__name__)


def format_quality_table(profiles: list, citation_map: dict = None) -> str:
    """
    Format paper quality profiles into a clean, reviewer-ready Markdown table
    displaying dual OpenAlex and Crossref identity checks.
    """
    if not profiles:
        return "No academic sources were available for quality auditing."

    lines = [
        "| # | Paper Title | OpenAlex | Crossref | Retraction Status | Journal / Venue | ISSN | Identity Verification |",
        "|---|-------------|----------|----------|-------------------|-----------------|------|-----------------------|",
    ]

    for i, p in enumerate(profiles, start=1):
        title = (p.get("title") or "Untitled")[:42]
        if len(p.get("title", "")) > 42:
            title += "..."

        url = (p.get("url") or "").strip().lower()
        cit_label = f"[{citation_map[url]}]" if citation_map and url in citation_map else f"[{i}]"

        # OpenAlex status badge
        oa_stat = p.get("openalex_status", "SKIPPED")
        if oa_stat == "VERIFIED":
            oa_badge = "✅ Verified"
        elif oa_stat == "NOT_FOUND":
            oa_badge = "❌ Not Found"
        elif oa_stat == "TIMEOUT":
            oa_badge = "⏳ Timeout"
        elif "Preprint" in oa_stat or "SKIPPED" in oa_stat:
            oa_badge = "⚪ Preprint"
        else:
            oa_badge = "⚠️ Unavailable"

        # Crossref status badge
        cr_stat = p.get("crossref_status", "SKIPPED")
        if cr_stat == "VERIFIED":
            cr_badge = "✅ Verified"
        elif cr_stat == "NOT_FOUND":
            cr_badge = "❌ Not Found"
        elif cr_stat == "TIMEOUT":
            cr_badge = "⏳ Timeout"
        elif "Preprint" in cr_stat or "SKIPPED" in cr_stat:
            cr_badge = "⚪ Preprint"
        else:
            cr_badge = "⚠️ Unavailable"

        if p.get("is_retracted"):
            retraction_status = "🔴 **RETRACTED**"
        else:
            retraction_status = "🟢 No retraction flag (OpenAlex/Crossref)"

        raw_venue = (p.get("journal_name") or p.get("venue") or p.get("publisher") or "Academic Record").strip()
        if raw_venue.lower() in ("unknown", "", "none"):
            raw_venue = p.get("publisher") or "Academic Record"
        venue = raw_venue[:45]
        if len(raw_venue) > 45:
            venue += "..."
        issn = p.get("issn") or "N/A"

        conf = p.get("identity_confidence") or p.get("confidence_level", "LOW")
        if conf == "HIGH":
            conf_badge = "🟢 HIGH"
        elif conf == "MEDIUM":
            conf_badge = "🟡 MEDIUM"
        elif conf == "RETRACTED_WARNING":
            conf_badge = "🔴 RETRACTED"
        else:
            conf_badge = "⚪ LOW"

        lines.append(
            f"| {cit_label} | {title} | {oa_badge} | {cr_badge} | {retraction_status} | {venue} | `{issn}` | {conf_badge} |"
        )

    lines.append("\n*> **Note on Scopus / Web of Science:** Not independently verified by our system (requires institutional enterprise API credentials).*")
    return "\n".join(lines)


def source_quality_agent(literature_sources: list) -> dict:
    """
    Perform a complete source quality verification for all literature sources.

    Returns
    -------
    dict:
      - 'profiles': list of PaperQualityProfile
      - 'markdown_table': table for final report
      - 'retracted_count': number of retracted papers detected
      - 'retraction_warnings': list of critical warnings
      - 'enriched_sources': literature_sources updated with verified metadata
    """
    if not literature_sources:
        return {
            "profiles": [],
            "markdown_table": "No academic literature sources to audit.",
            "retracted_count": 0,
            "retraction_warnings": [],
            "enriched_sources": [],
        }

    logger.info("[Source Quality Agent] Auditing %d academic papers...", len(literature_sources))
    profiles = batch_verify_papers(literature_sources)

    retraction_warnings = []
    enriched_sources = []

    for source, profile in zip(literature_sources, profiles):
        # Enrich source dictionary with verified details
        source["doi_verified"] = profile.get("doi_verified", False)
        source["is_retracted"] = profile.get("is_retracted", False)
        source["journal_name"] = profile.get("journal_name", source.get("venue", ""))
        source["venue"] = profile.get("journal_name", source.get("venue", ""))
        source["issn"] = profile.get("issn", "")
        source["openalex_status"] = profile.get("openalex_status", "SKIPPED")
        source["crossref_status"] = profile.get("crossref_status", "SKIPPED")
        source["peer_reviewed"] = profile.get("peer_reviewed", "Unknown / Unverified")
        source["identity_confidence"] = profile.get("identity_confidence", "LOW")
        source["confidence_level"] = source["identity_confidence"]
        source["verified_indexes"] = profile.get("indexed_in", [])
        source["citation_count"] = profile.get("citation_count", 0)

        if profile.get("is_retracted"):
            warning = f"CRITICAL: Paper '[{source.get('title')}]' (DOI: {source.get('doi')}) has been officially RETRACTED."
            retraction_warnings.append(warning)
            logger.warning("[Source Quality Alert] %s", warning)

        enriched_sources.append(source)

    doi_verified_count = sum(1 for p in profiles if p.get("doi_verified"))
    retracted_count = sum(1 for p in profiles if p.get("is_retracted"))
    high_conf_count = sum(1 for p in profiles if (p.get("identity_confidence") == "HIGH" or p.get("confidence_level") == "HIGH"))

    logger.info(
        "[Source Quality Agent] Audit complete. DOI Verified: %d/%d | High Confidence: %d | Retracted: %d",
        doi_verified_count, len(profiles), high_conf_count, retracted_count
    )

    return {
        "profiles": profiles,
        "markdown_table": format_quality_table(profiles),
        "retracted_count": retracted_count,
        "retraction_warnings": retraction_warnings,
        "enriched_sources": enriched_sources,
    }
