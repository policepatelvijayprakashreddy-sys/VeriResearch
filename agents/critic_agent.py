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
            "A concise search query of 3 to 6 keywords for "
            "the next research round, if more research is required. "
            "Must be short keywords only, NO full sentences, "
            "NO parentheses, and NO quotes."
        )
    )


from llm_client import create_llm


def critic_agent(
    topic,
    research_result,
    literature_analysis,
    literature_sources,
    evidence=None,
    source_quality_table="",
    retraction_warnings=None,
    consensus_table="",
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

    retraction_alert_text = "\n".join(f"- {w}" for w in (retraction_warnings or []))
    if not retraction_alert_text:
        retraction_alert_text = "No retracted papers were detected."

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

SOURCE QUALITY & RETRACTION AUDIT:
{source_quality_table or 'No source quality audit available.'}

RETRACTION ALERTS:
{retraction_alert_text}

EXTRACTED EVIDENCE:
{evidence_information}

CROSS-PAPER SCIENTIFIC CONSENSUS AUDIT:
{consensus_table or 'No cross-paper consensus audit available.'}


Evaluate the research carefully.


## 1. SOURCE QUALITY & RETRACTIONS

Review the Source Quality & Retraction Audit above:
- If ANY paper is flagged as RETRACTED, treat its claims as invalid and recommend more research if critical gaps remain.
- Check whether sources are verified in official registries (DOI, ISSN, PubMed, DOAJ).
- Distinguish between peer-reviewed journal papers and unvetted preprints.

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


## 5. SCIENTIFIC CONSENSUS & CONTRADICTIONS

Review the Cross-Paper Scientific Consensus Audit:
- Distinguish between a missing information gap and a GENUINE SCIENTIFIC DISAGREEMENT (MIXED_EVIDENCE).
- If the consensus verdict is MIXED_EVIDENCE, this indicates legitimate conflicting empirical findings across the literature under differing conditions. Do NOT request MORE_RESEARCH simply because literature evidence is mixed; report the nuance and differing scopes.
- Identify contradictions between sources only when they are actually supported by the provided information. Do not invent contradictions.


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
- follow_up_query must be a concise keyword query of 3 to 6 keywords (e.g. 'UAV localization CNN architectures'). Do NOT write a full sentence. Do NOT use parentheses, quotes, or boolean operators (AND/OR).

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