import sys
from pathlib import Path

# Ensure project root is in sys.path when running script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel, Field
from typing import Literal


class CriticOutput(BaseModel):
    decision: Literal[
        "MORE_RESEARCH",
        "SUFFICIENT"
    ]

    reason: str = Field(
        description="Why this decision was made."
    )

    knowledge_gap: str = Field(
        description=(
            "The most important missing "
            "information, if any."
        )
    )

    follow_up_query: str = Field(
        description=(
            "A specific search query for "
            "the next research round, "
            "if more research is required."
        )
    )


from llm_client import create_llm


def critic_agent(
    topic,
    research_result,
    literature_analysis,
    literature_sources,
    evidence=None
):
    """Evaluate research quality and decide whether more research is needed."""

    llm = create_llm()
    structured_llm = llm.with_structured_output(
        CriticOutput
    )

    # =========================================
    # SOURCE INFORMATION
    # =========================================

    if literature_sources:

        source_information = "\n\n".join(
            f"SOURCE {i + 1}\n"
            f"Title: {source['title']}\n"
            f"URL: {source['url']}\n"
            f"Search description: {source['content']}"
            for i, source in enumerate(
                literature_sources
            )
        )

    else:

        source_information = (
            "NO VERIFIED ACADEMIC SOURCES WERE FOUND."
        )

    evidence_information = "\n\n".join(
        f"SOURCE: {item.get('source_id', '')}\n"
        f"TITLE: {item.get('source_title', '')}\n"
        f"EVIDENCE: {item.get('evidence', '')}\n"
        f"CLAIM: {item.get('claim', '')}"
        for item in (evidence or [])
    )

    if not evidence_information:
        evidence_information = "NO EXTRACTED EVIDENCE WAS FOUND."

    # =========================================
    # CRITIC PROMPT
    # =========================================

    prompt = f"""
You are the verification and quality-control agent
for an AI research system.

Research topic:
{topic}

GENERAL RESEARCH:
{research_result}

LITERATURE ANALYSIS:
{literature_analysis}

ACTUAL LITERATURE SOURCES:
{source_information}

EXTRACTED EVIDENCE:
{evidence_information}


Evaluate the research carefully.


## 1. SOURCE QUALITY

For each literature source determine whether it appears
to be:

- an individual academic paper
- a publisher page
- a database/search page
- a general website
- another type of source

Do not assume that an academic domain automatically
means the specific result is an academic paper.


## 2. MISSING INFORMATION

Identify missing:

- authors
- publication years
- DOI/identifier
- findings
- limitations


## 3. RESEARCH QUALITY

Identify important weaknesses in the overall research.

Consider:

- factual support
- source quality
- breadth of coverage
- reliability
- missing evidence
- unsupported claims


## 4. KNOWLEDGE GAPS

Identify important topics that still need investigation.


## 5. CONTRADICTIONS

Identify contradictions between sources only when
they are actually supported by the provided information.

Do not invent contradictions.


## 6. SHOULD MORE RESEARCH BE PERFORMED?

Answer this conservatively.

Choose MORE_RESEARCH only if there is a significant
and specific information gap that another search is
likely to address.

Choose SUFFICIENT if:

- the main research questions are adequately covered
- additional searching is unlikely to add important information
- search results are unavailable
- the available evidence is already sufficient for a
  reasonable research report

Do NOT request another research round simply because
some information is imperfect.


## 7. NEXT RESEARCH ACTION

If MORE_RESEARCH is selected, describe the most important
research task that should happen next.

If SUFFICIENT is selected, explain why additional research
is not currently necessary.


IMPORTANT:

Do not invent facts.

If something cannot be determined, say:

"Cannot be determined from the available information."

Return the following information:

- decision
- reason
- knowledge_gap
- follow_up_query

The decision must be either:

MORE_RESEARCH

or:

SUFFICIENT

If the decision is SUFFICIENT:

- knowledge_gap should be an empty string
- follow_up_query should be an empty string

If the decision is MORE_RESEARCH:

- knowledge_gap must identify one specific missing research area
- follow_up_query must be a specific search query that investigates that gap

Do not invent facts.
Do not invent sources.
"""

    # If the LLM call fails outright, stop additional research safely.
    default_critique = CriticOutput(
        decision="SUFFICIENT",
        reason=(
            "Critic evaluation could not be generated due to an "
            "LLM error. Proceeding with available research."
        ),
        knowledge_gap="",
        follow_up_query=""
    )

    try:

        response = structured_llm.invoke(
            prompt
        )

        return response

    except Exception as error:

        print(
            f"[Critic Agent] Structured LLM call failed: {error}"
        )

        return default_critique