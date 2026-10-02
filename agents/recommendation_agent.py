"""
Recommendation Agent
====================

Identifies under-explored research directions within a topic.

Pipeline
--------
1. Run a targeted gap-discovery search (e.g. "open problems in <topic>")
2. Combine those results with already-gathered literature sources
3. Ask the LLM to produce 5–8 structured research project ideas

Each recommendation includes:
  - title      : specific project name
  - gap        : why this area is under-explored
  - approach   : suggested methodology
  - novelty    : LOW | MEDIUM | HIGH
  - difficulty : STARTER | INTERMEDIATE | ADVANCED

Usage
-----
    from agents.recommendation_agent import recommendation_agent

    recs = recommendation_agent(topic, literature_sources, web_sources, literature_analysis="", verified_evidence=[])
    for r in recs:
        print(r.title, r.novelty)
"""

import logging
import sys
from pathlib import Path
from typing import List, Literal

# Ensure project root is in sys.path when running script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel, Field

from llm_client import create_llm
from search.search_manager import search

logger = logging.getLogger(__name__)

# Max gap-discovery search results to feed in
_GAP_SEARCH_MAX = 6


# ============================================================
# PYDANTIC MODELS
# ============================================================

class ResearchRecommendation(BaseModel):
    """One under-explored research project idea."""

    title: str = Field(
        description=(
            "A specific, descriptive title for the proposed research project. "
            "Should be concrete, not generic."
        )
    )

    gap: str = Field(
        description=(
            "Why this area is under-explored or under-studied. "
            "Reference evidence from the provided sources where possible."
        )
    )

    approach: str = Field(
        description=(
            "Suggested methodology or research approach for tackling this gap."
        )
    )

    novelty: Literal["LOW", "MEDIUM", "HIGH"] = Field(
        description=(
            "Estimated novelty relative to existing literature. "
            "HIGH = genuinely new direction; LOW = incremental extension."
        )
    )

    difficulty: Literal["STARTER", "INTERMEDIATE", "ADVANCED"] = Field(
        description=(
            "Estimated difficulty for a research team. "
            "STARTER = achievable in a few months; ADVANCED = multi-year effort."
        )
    )


class RecommendationOutput(BaseModel):
    """Structured list of research project recommendations."""

    recommendations: List[ResearchRecommendation] = Field(
        description="5 to 8 under-explored research project ideas."
    )

    landscape_summary: str = Field(
        description=(
            "A 2–3 sentence summary of the current research landscape "
            "that contextualises the recommended gaps."
        )
    )


# ============================================================
# GAP DISCOVERY SEARCH
# ============================================================

def _gap_search(topic: str) -> list:
    """
    Run a targeted search to discover known open problems
    and under-explored directions in the topic area.
    """

    queries = [
        f"open problems unsolved challenges {topic}",
        f"future research directions gaps {topic}",
    ]

    results = []
    seen_urls: set = set()

    for q in queries:

        logger.info("[Recommendation] Gap search: %s", q)

        hits = search(q, max_results=_GAP_SEARCH_MAX)

        for hit in hits:
            url = hit.get("url", "").strip().lower()
            if url and url not in seen_urls:
                seen_urls.add(url)
                results.append(hit)

    logger.info(
        "[Recommendation] Gap search found %d unique sources.", len(results)
    )

    return results


# ============================================================
# FORMAT SOURCES FOR PROMPT
# ============================================================

def _format_sources(web_sources: list, academic_sources: list, gap_sources: list) -> str:
    """Combine and format all sources for the LLM prompt."""

    sections = []

    if academic_sources:
        acad_text = "\n\n".join(
            f"ACADEMIC [{i+1}]\n"
            f"Title: {s.get('title', '')}\n"
            f"Authors: {', '.join(s.get('authors', [])) or 'N/A'}\n"
            f"Year: {s.get('year', 'N/A')}\n"
            f"Abstract: {s.get('content', '')[:400]}"
            for i, s in enumerate(academic_sources[:8])
        )
        sections.append(f"=== ACADEMIC LITERATURE ===\n\n{acad_text}")

    if web_sources:
        web_text = "\n\n".join(
            f"WEB [{i+1}]\n"
            f"Title: {s.get('title', '')}\n"
            f"URL: {s.get('url', '')}\n"
            f"Snippet: {s.get('content', '')[:300]}"
            for i, s in enumerate(web_sources[:6])
        )
        sections.append(f"=== WEB RESEARCH ===\n\n{web_text}")

    if gap_sources:
        gap_text = "\n\n".join(
            f"GAP SOURCE [{i+1}]\n"
            f"Title: {s.get('title', '')}\n"
            f"Snippet: {s.get('content', '')[:400]}"
            for i, s in enumerate(gap_sources[:6])
        )
        sections.append(f"=== OPEN PROBLEMS / GAPS FOUND ===\n\n{gap_text}")

    return "\n\n".join(sections) if sections else "No sources available."


def _token_jaccard(a: str, b: str) -> float:
    import re
    sa = set(re.findall(r"\w+", (a or "").lower()))
    sb = set(re.findall(r"\w+", (b or "").lower()))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _is_valid_rec(r: ResearchRecommendation, source_titles: list) -> bool:
    gap_words = (r.gap or "").split()
    app_words = (r.approach or "").split()
    if len(gap_words) < 8 or "Novelty =" in r.gap or "Difficulty =" in r.gap or len(app_words) < 8:
        return False
    if any(_token_jaccard(r.title, t) > 0.65 for t in source_titles if t):
        return False
    return True


# ============================================================
# MAIN RECOMMENDATION AGENT
# ============================================================

def recommendation_agent(
    topic: str,
    literature_sources: list,
    web_sources: list,
    literature_analysis: str = "",
    verified_evidence: list = None,
) -> RecommendationOutput:
    """
    Identify 5–8 under-explored research directions for a given topic.

    Parameters
    ----------
    topic : str
        The research topic entered by the user.
    literature_sources : list
        Academic sources already gathered by the literature agent.
    web_sources : list
        Web sources already gathered by the research agent.
    literature_analysis : str
        The narrative literature review analysis.
    verified_evidence : list
        The list of claims verified by the NLI agent.

    Returns
    -------
    RecommendationOutput
        Structured list of ResearchRecommendation objects, plus a
        landscape summary.  On failure, returns an object with an
        empty list and an error message as the summary.
    """

    logger.info("[Recommendation] Generating research project ideas for: %s", topic)

    # ── 1. Gap discovery search ───────────────────────────────
    gap_sources = _gap_search(topic)

    # ── 2. Build prompt context ───────────────────────────────
    source_context = _format_sources(web_sources, literature_sources, gap_sources)
    verified_evidence = verified_evidence or []
    
    evidence_text = ""
    if verified_evidence:
        evidence_text = "=== VERIFIED CLAIMS EXTRACTED FROM SOURCES ===\n\n"
        for i, ev in enumerate(verified_evidence[:10]):
            evidence_text += f"- [{ev.get('label', 'UNKNOWN')}] {ev.get('claim', '')}\n"

    prompt = f"""You are an expert research strategist specialising in identifying
under-explored opportunities within academic fields.

Research topic: {topic}

AVAILABLE SOURCES:

{source_context}

{evidence_text}

=== PREVIOUS LITERATURE ANALYSIS ===
{literature_analysis if literature_analysis else 'None provided.'}

======================================
TASK
======================================

Based on the sources above, identify 5 to 8 specific research projects
that are UNDER-EXPLORED or represent genuine GAPS in the current literature.

Rules:
- Ground each recommendation in the provided sources where possible.
- Do NOT recommend topics that are already extensively covered.
- Do NOT copy the exact title of an existing paper in the sources as a project title.
- gap: MUST be a 2–3 sentence technical explanation of why this specific problem is unsolved (e.g., lack of sensor datasets, computational complexity, unverified assumptions). Do NOT repeat metadata strings like 'Novelty = HIGH' in the gap field.
- approach: MUST be a 2–3 sentence technical methodology explaining how a researcher should solve the problem.
- DIVERSE METHODOLOGIES: Ensure the recommended projects use varied technical approaches (e.g., benchmark suite construction, formal verification, architectural ablation, security red-teaming, or empirical system evaluation).
- DIVERSE COMPLEXITY: Include at least one ADVANCED multi-year technical project, at least one INTERMEDIATE, and at least one STARTER project.
- novelty: HIGH (genuinely new direction), MEDIUM (significant advance), LOW (incremental extension).
- difficulty: STARTER (few months), INTERMEDIATE (6-12 months), ADVANCED (multi-year).

Also write a 2–3 sentence landscape_summary explaining the current state of research on this topic and why these gaps exist.
"""

    # ── 3. Structured LLM call ───────────────────────────────
    llm = create_llm()
    structured_llm = llm.with_structured_output(RecommendationOutput)

    fallback = RecommendationOutput(
        recommendations=[],
        landscape_summary=(
            "Recommendation generation failed due to an LLM error. "
            "Please retry with Ollama running."
        ),
    )

    try:
        result = structured_llm.invoke(prompt)
        all_titles = [s.get("title", "") for s in (literature_sources + web_sources + gap_sources) if s.get("title")]
        valid_recs = [r for r in result.recommendations if _is_valid_rec(r, all_titles)]
        result.recommendations = valid_recs
        logger.info(
            "[Recommendation] Generated %d valid recommendations (out of %d raw).",
            len(valid_recs),
            len(result.recommendations)
        )
        return result

    except Exception as error:
        logger.error("[Recommendation] LLM call failed: %s", error)
        return fallback


# ============================================================
# FORMAT FOR REPORT APPENDIX
# ============================================================

def format_recommendations_md(output: RecommendationOutput, topic: str) -> str:
    """
    Format the recommendations as a markdown section to append
    to the research report.
    """

    if not output.recommendations:
        return (
            "\n\n---\n\n"
            "## Research Opportunities\n\n"
            "*No recommendations could be generated.*\n"
        )

    novelty_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
    difficulty_emoji = {
        "STARTER": "⭐",
        "INTERMEDIATE": "⭐⭐",
        "ADVANCED": "⭐⭐⭐",
    }

    lines = [
        "\n\n---\n",
        "## Research Opportunities\n",
        f"> *Under-explored directions in: **{topic}***\n",
        f"{output.landscape_summary}\n",
    ]

    for i, rec in enumerate(output.recommendations, start=1):
        n_icon = novelty_emoji.get(rec.novelty, "")
        d_icon = difficulty_emoji.get(rec.difficulty, "")
        lines += [
            f"### {i}. {rec.title}",
            f"\n**Novelty:** {n_icon} {rec.novelty} &nbsp;|&nbsp; "
            f"**Difficulty:** {d_icon} {rec.difficulty}\n",
            f"**Why under-explored:** {rec.gap}\n",
            f"**Suggested approach:** {rec.approach}\n",
        ]

    return "\n".join(lines)


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    import sys

    topic = " ".join(sys.argv[1:]) or "federated learning in healthcare"

    result = recommendation_agent(
        topic=topic,
        literature_sources=[],
        web_sources=[],
    )

    print(f"\nLandscape: {result.landscape_summary}\n")

    for i, r in enumerate(result.recommendations, 1):
        print(f"{i}. [{r.novelty}] {r.title}")
        print(f"   Gap: {r.gap[:120]}...")
        print(f"   Approach: {r.approach[:120]}...")
        print()
