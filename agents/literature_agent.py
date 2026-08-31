import logging

from search.academic_search_manager import academic_search

from config import (
    LITERATURE_SEARCH_MAX_RESULTS,
    MAX_ACADEMIC_SOURCES,
)

from llm_client import create_llm

logger = logging.getLogger(__name__)


# ============================================================
# LITERATURE SEARCH
# ============================================================

def search_literature(
    topic,
    max_results=LITERATURE_SEARCH_MAX_RESULTS,
):
    """
    Search specifically for academic literature.

    Uses:

        Semantic Scholar
              ↓
        Crossref fallback
              ↓
        normalized academic results
    """

    print(
        "\n[Literature Search]"
    )

    print(
        f"Research topic: {topic}"
    )

    results = academic_search(
        topic,
        max_results=max_results,
    )

    if not results:

        print(
            "No academic literature found."
        )

        return []

    # Keep the number of sources
    # within the project limit.

    results = results[
        :MAX_ACADEMIC_SOURCES
    ]

    print(
        f"Using {len(results)} "
        f"academic sources."
    )

    return results


# ============================================================
# FORMAT SOURCES
# ============================================================

def format_sources(
    literature_sources
):
    """
    Convert academic sources into text
    that can be given to the LLM.
    """

    if not literature_sources:

        return (
            "NO VERIFIED ACADEMIC SOURCES "
            "WERE FOUND."
        )

    formatted = []

    for i, source in enumerate(
        literature_sources,
        start=1
    ):

        authors = source.get(
            "authors",
            []
        )

        if authors:

            author_text = ", ".join(
                authors
            )

        else:

            author_text = (
                "Not available"
            )

        year = source.get(
            "year"
        )

        if year is None:

            year_text = (
                "Not available"
            )

        else:

            year_text = str(
                year
            )

        doi = source.get(
            "doi",
            ""
        )

        if not doi:

            doi = (
                "Not available"
            )

        formatted.append(
            f"""
SOURCE {i}

Title:
{source.get('title', 'Untitled')}

Authors:
{author_text}

Publication year:
{year_text}

DOI:
{doi}

URL:
{source.get('url', '')}

Academic provider:
{source.get('source', 'Unknown')}

Abstract:
{source.get('content', '')}
"""
        )

    return "\n".join(
        formatted
    )


# ============================================================
# LITERATURE ANALYSIS
# ============================================================

def analyze_literature(
    topic,
    literature_sources,
):
    """
    Analyze the academic literature found
    by the Academic Search Manager.
    """

    llm = create_llm()

    source_information = format_sources(
        literature_sources
    )

    prompt = f"""
You are an academic literature analysis agent.

Research topic:
{topic}

ACADEMIC SOURCES:

{source_information}


Analyze the literature carefully.

For each important source discuss:

1. Research focus
2. Main findings, if available
3. Limitations, if available
4. Relevance to the research topic


Also identify:

- common themes
- differences between sources
- missing information
- knowledge gaps
- weaknesses in the available literature


IMPORTANT RULES:

- Use ONLY the information provided.
- Do not invent authors.
- Do not invent publication years.
- Do not invent DOI numbers.
- Do not invent findings.
- Do not invent limitations.
- Do not assume an article is a research paper
  simply because it has a DOI.
- If information is unavailable, say:

"Not enough information was available."

Do not create additional sources.

Do not create a bibliography.

Return ONLY the literature analysis.
"""

    try:

        response = llm.invoke(prompt)

        return response.content.strip()

    except Exception as error:

        logger.error(
            "[Literature Agent] LLM call failed: %s", error
        )

        return ""


# ============================================================
# MAIN LITERATURE AGENT
# ============================================================

def literature_agent(
    topic
):
    """
    Complete literature research pipeline.

        Topic
          ↓
    Academic Search Manager
          ↓
    Semantic Scholar
          ↓
    Crossref fallback
          ↓
    Academic sources
          ↓
    Literature analysis
    """

    literature_sources = search_literature(
        topic
    )

    literature_analysis = analyze_literature(
        topic,
        literature_sources
    )

    return {
        "analysis": literature_analysis,

        "sources": literature_sources,

        "source_count": len(
            literature_sources
        ),
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    topic = (
        "artificial intelligence "
        "healthcare"
    )

    result = literature_agent(
        topic
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "LITERATURE SOURCES"
    )

    print(
        "=" * 60
    )

    for i, source in enumerate(
        result["sources"],
        start=1
    ):

        print(
            f"\n{i}. "
            f"{source['title']}"
        )

        print(
            f"Authors: "
            f"{', '.join(source['authors'])}"
        )

        print(
            f"Year: "
            f"{source['year']}"
        )

        print(
            f"DOI: "
            f"{source['doi']}"
        )

        print(
            f"URL: "
            f"{source['url']}"
        )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "LITERATURE ANALYSIS"
    )

    print(
        "=" * 60
    )

    print(
        result["analysis"]
    )