import logging
import sys
from pathlib import Path
from typing import List

# Ensure project root is in sys.path when running script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel, Field

from llm_client import create_llm

logger = logging.getLogger(__name__)


class EvidenceItem(BaseModel):
    source_id: str = Field(
        description="The source URL or a stable source identifier."
    )

    source_title: str = Field(
        description="The title of the source."
    )

    source_url: str = Field(
        description="The exact URL of the source."
    )

    evidence: str = Field(
        description="Directly supported evidence from the source content."
    )

    claim: str = Field(
        description="A concise research claim supported by the evidence."
    )


class EvidenceOutput(BaseModel):
    items: List[EvidenceItem] = Field(
        description="Evidence items extracted from the supplied sources."
    )


def extract_evidence(topic, sources):
    """Extract source-grounded evidence for a research topic."""

    source_information = "\n\n".join(
        f"SOURCE_ID: source_{i + 1}\n"
        f"TITLE: {source.get('title', '')}\n"
        f"URL: {source.get('url', '')}\n"
        f"CONTENT: {source.get('content', '')}"
        for i, source in enumerate(sources)
    )

    prompt = f"""
You are an evidence extraction agent.

Research topic:
{topic}

Sources:
{source_information or 'No sources were provided.'}

Extract only evidence directly supported by the source content.
For each useful source, provide:

- source_id: MUST strictly be the exact SOURCE_ID (e.g. source_1, source_2) corresponding to the CONTENT where the quote was taken.
- source_title: the source title
- source_url: the exact source URL
- evidence: MUST be an exact quote or verbatim passage copied directly from CONTENT. Do not paraphrase or use titles as evidence.
- claim: a clear, concise claim (at least 6 words) supported by that evidence. Never leave blank.

Do not create or modify URLs.
Do not invent facts, sources, authors, dates, statistics, or conclusions.
Do not include evidence that is not present in the supplied source content.
"""

    llm = create_llm()

    structured_llm = llm.with_structured_output(
        EvidenceOutput
    )

    try:

        response = structured_llm.invoke(prompt)

        return response

    except Exception as error:

        logger.error(
            "[Evidence Agent] Structured LLM call failed: %s", error
        )

        return EvidenceOutput(items=[])


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_sources = [
        {
            "title": (
                "Artificial intelligence "
                "in healthcare"
            ),
            "url": "https://example.com/paper",
            "content": (
                "Artificial intelligence "
                "can assist healthcare "
                "professionals with diagnostic "
                "tasks."
            ),
        }
    ]

    result = extract_evidence(
        "Artificial Intelligence in Healthcare",
        test_sources,
    )

    print(
        "\n=============================="
    )

    print(
        "EXTRACTED EVIDENCE"
    )

    print(
        "=============================="
    )

    for item in result.items:

        print(
            f"\nSource: "
            f"{item.source_id}"
        )

        print(
            f"Title: "
            f"{item.source_title}"
        )

        print(
            f"Evidence: "
            f"{item.evidence}"
        )

        print(
            f"Claim: "
            f"{item.claim}"
        )
