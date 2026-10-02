import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path when running script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from search.search_manager import search
from llm_client import create_llm, safe_invoke
from config import (
    RESEARCH_INITIAL_MAX_RESULTS,
    RESEARCH_FOLLOWUP_MAX_RESULTS,
)

logger = logging.getLogger(__name__)


def summarize_research(
    llm,
    topic,
    results,
    previous_summary=""
):
    """Create or update the research summary."""

    if not results:
        return previous_summary

    context = "\n\n".join(
        f"TITLE: {r.get('title', '')}\n"
        f"URL: {r.get('url', '')}\n"
        f"CONTENT: {r.get('content', '')}"
        for r in results
    )

    prompt = f"""
You are a research assistant.

Research topic:
{topic}

Previous research summary:
{previous_summary}

New web research:
{context}

Create an accurate updated research summary.

Rules:

- Use only information supported by the provided research.
- Do not invent facts.
- Do not invent sources.
- Do not invent statistics.
- Do not invent authors or publication dates.
- If information is uncertain, say so.
- Incorporate useful new information into the previous summary.
- Avoid unnecessary repetition.

Return only the research summary.
"""

    return safe_invoke(
        llm,
        prompt,
        default=previous_summary
    )


def find_knowledge_gap(
    llm,
    topic,
    summary
):
    """Identify one important missing research area."""

    prompt = f"""
You are a critical research assistant.

Research topic:
{topic}

Current research summary:
{summary}

Identify ONE important knowledge gap.

Then create ONE concise search query (3 to 6 keywords only) that would
specifically help investigate that gap. Do NOT write a complete sentence.
Do NOT use parentheses, quotes, or boolean operators (AND/OR).

Return exactly:

GAP: <specific missing information>
QUERY: <concise 3-6 keywords search query>
"""

    text = safe_invoke(
        llm,
        prompt,
        default=""
    )

    gap = ""
    query = ""

    for line in text.splitlines():

        if line.startswith("GAP:"):
            gap = line.replace(
                "GAP:",
                "",
                1
            ).strip()

        elif line.startswith("QUERY:"):
            query = line.replace(
                "QUERY:",
                "",
                1
            ).strip()

    # If the LLM generated an overly short or detached query (e.g. "Chain"),
    # anchor it with the core topic keywords to prevent off-topic search queries.
    if query:
        query_words = query.split()
        if len(query_words) <= 2:
            topic_keywords = " ".join([w for w in topic.split() if len(w) > 3][:3])
            query = f"{topic_keywords} {query}"

    return gap, query


def generate_initial_query(
    llm,
    topic
):
    """Generate the initial research query."""

    prompt = f"""
Create ONE high-quality web search query (3 to 6 keywords)
for the following research topic:

{topic}

The query should focus on finding useful, reliable research information.
Keep it short (3 to 6 keywords). Do NOT write a sentence. Do NOT use quotes or parentheses.

Return ONLY the search query keywords.
"""

    # Fall back to the raw topic itself if the LLM call fails,
    # so a downed Ollama instance doesn't kill the search step.
    return safe_invoke(
        llm,
        prompt,
        default=topic
    )


def merge_sources(
    previous_sources,
    new_sources
):
    """Merge sources and remove duplicate URLs."""

    combined = []

    seen_urls = set()

    for source in (
        previous_sources + new_sources
    ):

        url = source.get(
            "url",
            ""
        ).strip().lower()

        if not url:
            continue

        if url in seen_urls:
            continue

        seen_urls.add(url)

        combined.append(source)

    return combined


def research_agent(
    topic,
    previous_research="",
    follow_up_query=""
):
    """
    Perform web research and return both the
    summary and the actual sources.
    """

    llm = create_llm()

    # =========================================
    # INITIAL QUERY
    # =========================================

    if follow_up_query and follow_up_query.strip():
        query = follow_up_query.strip()
    else:
        query = generate_initial_query(
            llm,
            topic
        )

    logger.info("[Research Search]")
    logger.info("Search query: %s", query)

    # =========================================
    # ROUND 1 SEARCH
    # =========================================

    results = search(
        query,
        max_results=RESEARCH_INITIAL_MAX_RESULTS
    )

    logger.info("Found %d sources.", len(results))

    # =========================================
    # SEARCH FAILURE
    # =========================================

    if not results:

        logger.warning("Web search failed. Keeping previous research.")

        return {
            "summary": previous_research,
            "sources": [],
            "new_sources_found": False,
            "knowledge_gap": "",
            "follow_up_query": ""
        }

    # =========================================
    # UPDATE SUMMARY
    # =========================================

    summary = summarize_research(
        llm,
        topic,
        results,
        previous_summary=previous_research
    )

    # =========================================
    # REFLECTION
    # =========================================

    logger.info("[Reflection]")

    gap, follow_up_query = find_knowledge_gap(
        llm,
        topic,
        summary
    )

    logger.info("Knowledge gap: %s", gap)
    logger.info("Follow-up query: %s", follow_up_query)

    # =========================================
    # FOLLOW-UP SEARCH
    # =========================================

    follow_up_results = []

    if follow_up_query:

        logger.info("[Research Follow-up]")

        follow_up_results = search(
            follow_up_query,
            max_results=RESEARCH_FOLLOWUP_MAX_RESULTS
        )

        logger.info(
            "Found %d additional sources.", len(follow_up_results)
        )

        if follow_up_results:

            summary = summarize_research(
                llm,
                topic,
                follow_up_results,
                previous_summary=summary
            )

    # =========================================
    # COMBINE SOURCES
    # =========================================

    all_new_sources = (
        results
        + follow_up_results
    )

    unique_sources = merge_sources(
        [],
        all_new_sources
    )

    # =========================================
    # RETURN STRUCTURED RESULT
    # =========================================

    return {
        "summary": summary,

        "sources": unique_sources,

        "new_sources_found": bool(
            unique_sources
        ),

        "knowledge_gap": gap,

        "follow_up_query": follow_up_query
    }