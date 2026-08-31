import logging
from typing import TypedDict, Any

from langgraph.graph import StateGraph, START, END

from agents.research_agent import research_agent
from agents.literature_agent import literature_agent
from agents.evidence_agent import extract_evidence
from agents.critic_agent import critic_agent
from agents.report_agent import report_agent
from agents.recommendation_agent import (
    recommendation_agent,
    format_recommendations_md,
)
from agents.verification_agent import verification_agent

from config import (
    MAX_WEB_SOURCES,
    MAX_ACADEMIC_SOURCES,
    MAX_RESEARCH_ROUNDS,
)

logger = logging.getLogger(__name__)


# ============================================================
# GRAPH STATE
# ============================================================

class ResearchState(TypedDict, total=False):

    # User input
    topic: str

    # General web research
    research: str
    web_sources: list
    knowledge_gap: str
    follow_up_query: str

    # Academic research
    literature: dict

    # Extracted evidence
    evidence: list
    verification_results: list
    hallucination_rate: float

    # Critic
    critique: Any
    decision: str

    # Control
    research_round: int
    new_research_found: bool
    new_literature_found: bool

    # Final output
    final_report: str

    # Recommendations
    recommendations: list


# ============================================================
# RESEARCH NODE
# ============================================================

def research_node(state: ResearchState):

    current_round = (
        state.get("research_round", 0) + 1
    )

    logger.info(
        "Running Research Agent (Graph Round %d)...",
        current_round
    )

    # --------------------------------------------------------
    # Previous research
    # --------------------------------------------------------

    previous_research = state.get(
        "research",
        ""
    )

    # --------------------------------------------------------
    # Previous web sources
    # --------------------------------------------------------

    previous_web_sources = state.get(
        "web_sources",
        []
    )

    # --------------------------------------------------------
    # Run Research Agent
    # --------------------------------------------------------

    try:

        result = research_agent(
            state["topic"],
            previous_research
        )

    except Exception as error:

        print(
            f"\n[Research Node] "
            f"Research agent failed: {error}"
        )

        return {
            "research": previous_research,
            "web_sources": previous_web_sources,
            "knowledge_gap": state.get(
                "knowledge_gap",
                ""
            ),
            "follow_up_query": state.get(
                "follow_up_query",
                ""
            ),
            "research_round": current_round,
            "new_research_found": False
        }

    # --------------------------------------------------------
    # Read result safely
    # --------------------------------------------------------

    summary = result.get(
        "summary",
        previous_research
    )

    new_sources = result.get(
        "sources",
        []
    )

    new_sources_found = result.get(
        "new_sources_found",
        False
    )

    knowledge_gap = result.get(
        "knowledge_gap",
        ""
    )

    follow_up_query = result.get(
        "follow_up_query",
        ""
    )

    # --------------------------------------------------------
    # Combine previous + new web sources
    # --------------------------------------------------------

    combined_sources = (
        previous_web_sources
        + new_sources
    )

    # --------------------------------------------------------
    # Remove duplicate URLs
    # --------------------------------------------------------

    unique_sources = []

    seen_urls = set()

    for source in combined_sources:

        url = source.get(
            "url",
            ""
        ).strip().lower()

        if not url:
            continue

        if url in seen_urls:
            continue

        seen_urls.add(url)

        unique_sources.append(
            source
        )

    # --------------------------------------------------------
    # Limit general web sources
    # --------------------------------------------------------

    unique_sources = unique_sources[
        :MAX_WEB_SOURCES
    ]

    # --------------------------------------------------------
    # Display information
    # --------------------------------------------------------

    logger.info(
        "Total web sources kept: %d", len(unique_sources)
    )

    logger.info(
        "New web sources found: %s", new_sources_found
    )

    # --------------------------------------------------------
    # Return updated state
    # --------------------------------------------------------

    return {

        "research": summary,

        "web_sources": unique_sources,

        "knowledge_gap": knowledge_gap,

        "follow_up_query": follow_up_query,

        "research_round": current_round,

        "new_research_found": new_sources_found
    }


# ============================================================
# LITERATURE NODE
# ============================================================

def literature_node(state: ResearchState):

    logger.info("Running Literature Agent...")

    # --------------------------------------------------------
    # Previous literature
    # --------------------------------------------------------

    previous_literature = state.get(
        "literature",
        {}
    )

    previous_sources = previous_literature.get(
        "sources",
        []
    )

    # --------------------------------------------------------
    # Ask Literature Agent for NEW sources
    # --------------------------------------------------------

    try:

        result = literature_agent(
            state["topic"]
        )

    except Exception as error:

        print(
            f"\n[Literature Node] "
            f"Literature agent failed: {error}"
        )

        return {
            "literature": previous_literature,
            "new_literature_found": False
        }

    # --------------------------------------------------------
    # Read the current Literature Agent output contract
    # --------------------------------------------------------

    literature_analysis = result.get(
        "analysis",
        ""
    )

    literature_sources = result.get(
        "sources",
        []
    )

    new_sources = literature_sources

    # --------------------------------------------------------
    # Merge previous + new
    # --------------------------------------------------------

    combined_sources = (
        previous_sources
        + new_sources
    )

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    unique_sources = []

    seen_urls = set()

    for source in combined_sources:

        url = source.get(
            "url",
            ""
        ).strip().lower()

        if not url:
            continue

        if url in seen_urls:
            continue

        seen_urls.add(url)

        unique_sources.append(
            source
        )

    # --------------------------------------------------------
    # Limit total academic sources
    # --------------------------------------------------------

    unique_sources = unique_sources[
        :MAX_ACADEMIC_SOURCES
    ]

    # --------------------------------------------------------
    # Display information
    # --------------------------------------------------------

    logger.info(
        "Found %d new academic sources.", len(new_sources)
    )

    logger.info(
        "Total academic sources kept: %d", len(unique_sources)
    )

    # --------------------------------------------------------
    # Return accumulated literature
    # --------------------------------------------------------

    return {

        "literature": {

            "analysis": literature_analysis,

            "sources": unique_sources
        },

        "new_literature_found": bool(
            new_sources
        )
    }


# ============================================================
# EVIDENCE NODE
# ============================================================

def evidence_node(state: ResearchState):

    logger.info("Running Evidence Agent...")

    topic = state["topic"]

    web_sources = state.get(
        "web_sources",
        []
    )

    literature = state.get(
        "literature",
        {}
    )

    literature_sources = literature.get(
        "sources",
        []
    )

    all_sources = (
        web_sources
        + literature_sources
    )

    source_records = {
        f"source_{index}": source
        for index, source in enumerate(all_sources, start=1)
    }

    sources_by_url = {
        source.get("url", "").strip().lower(): (
            f"source_{index}", source
        )
        for index, source in enumerate(all_sources, start=1)
        if source.get("url", "").strip()
    }

    logger.info(
        "[Evidence Node] Processing %d sources.", len(all_sources)
    )

    if not all_sources:

        logger.info("[Evidence Node] No sources available.")

        return {
            "evidence": []
        }

    try:

        result = extract_evidence(
            topic,
            all_sources
        )

        evidence_items = []

        for item in result.items:
            source_id = item.source_id.strip()
            source = source_records.get(source_id)

            if source is None:
                matched_source = sources_by_url.get(
                    source_id.lower()
                )

                if matched_source:
                    source_id, source = matched_source

            if source is None:
                continue

            evidence_items.append({
                "source_id": source_id,
                "source_title": source.get(
                    "title",
                    item.source_title
                ),
                "source_url": source.get("url", ""),
                "evidence": item.evidence,
                "claim": item.claim,
            })

    except Exception as error:

        print(
            f"[Evidence Node] Evidence extraction failed: {error}"
        )

        return {
            "evidence": []
        }

    logger.info(
        "[Evidence Node] Extracted %d evidence items.",
        len(evidence_items)
    )

    return {
        "evidence": evidence_items
    }

# ============================================================
# VERIFICATION NODE
# ============================================================

def verification_node(state: ResearchState):
    logger.info("Running Verification Agent...")
    
    evidence_items = state.get("evidence", [])
    
    if not evidence_items:
        return {
            "verification_results": [],
            "hallucination_rate": None
        }
        
    try:
        summary = verification_agent(evidence_items)
        
        return {
            "verification_results": summary.get("results", []),
            "hallucination_rate": summary.get("hallucination_rate")
        }
        
    except Exception as error:
        logger.error("[Verification Node] Failed: %s", error)
        return {
            "verification_results": [],
            "hallucination_rate": None
        }

# ============================================================
# CRITIC NODE
# ============================================================

def critic_node(state: ResearchState):

    logger.info("Running Critic Agent...")

    literature = state.get(
        "literature",
        {}
    )

    literature_analysis = literature.get(
        "analysis",
        ""
    )

    literature_sources = literature.get(
        "sources",
        []
    )

    # --------------------------------------------------------
    # Run critic safely
    # --------------------------------------------------------

    try:

        result = critic_agent(
            state["topic"],
            state.get(
                "research",
                ""
            ),
            literature_analysis,
            literature_sources,
            state.get(
                "evidence",
                []
            )
        )

    except Exception as error:

        print(
            f"\n[Critic Node] "
            f"Critic agent failed: {error}"
        )

        # Safe fallback:
        # If the critic is unavailable, don't keep
        # researching forever.
        return {
            "critique": (
                "Critic agent failed. "
                "Proceeding with available evidence."
            ),
            "decision": "SUFFICIENT"
        }

    # --------------------------------------------------------
    # Extract decision
    # --------------------------------------------------------

    decision = getattr(
        result,
        "decision",
        ""
    )

    if decision not in {
        "MORE_RESEARCH",
        "SUFFICIENT"
    }:

        logger.warning("[Critic Warning] No valid decision was returned.")

        # Conservative stopping fallback
        decision = "SUFFICIENT"

    logger.info("[Critic Decision] %s", decision)

    return {
        "critique": result,
        "decision": decision
    }


# ============================================================
# ROUTER
# ============================================================

def route_after_critic(
    state: ResearchState
):

    decision = state.get(
        "decision",
        "SUFFICIENT"
    )

    research_round = state.get(
        "research_round",
        1
    )

    new_research = state.get(
        "new_research_found",
        False
    )

    new_literature = state.get(
        "new_literature_found",
        False
    )

    # ========================================================
    # MORE RESEARCH BUT NOTHING NEW WAS FOUND
    # ========================================================

    if (
        decision == "MORE_RESEARCH"
        and not new_research
        and not new_literature
    ):

        logger.info(
            "[Graph Decision] No new information was found. Stopping research."
        )

        return "report"

    # ========================================================
    # MORE RESEARCH + NEW INFORMATION
    # ========================================================

    if (
        decision == "MORE_RESEARCH"
        and (
            new_research
            or new_literature
        )
        and research_round < MAX_RESEARCH_ROUNDS
    ):

        logger.info(
            "[Graph Decision] More useful information may be found."
        )

        return "research"

    # ========================================================
    # MAXIMUM ROUNDS
    # ========================================================

    if (
        decision == "MORE_RESEARCH"
        and research_round > MAX_RESEARCH_ROUNDS
    ):

        logger.info("[Graph Decision] Maximum research rounds reached.")

        return "report"

    # ========================================================
    # SUFFICIENT
    # ========================================================

    logger.info("[Graph Decision] Research is sufficient.")

    return "report"


# ============================================================
# REPORT NODE
# ============================================================

def report_node(state: ResearchState):

    logger.info("Running Report Agent...")

    literature = state.get(
        "literature",
        {}
    )

    literature_analysis = literature.get(
        "analysis",
        ""
    )

    literature_sources = literature.get(
        "sources",
        []
    )

    web_sources = state.get(
        "web_sources",
        []
    )

    evidence = state.get(
        "evidence",
        []
    )

    # --------------------------------------------------------
    # Generate final report
    # --------------------------------------------------------

    try:

        result = report_agent(

            state["topic"],

            state.get(
                "research",
                ""
            ),

            web_sources,

            literature_analysis,

            literature_sources,

            state.get(
                "critique",
                ""
            ),

            evidence,
            
            state.get(
                "verification_results",
                []
            )
        )

    except Exception as error:

        print(
            f"\n[Report Node] "
            f"Report agent failed: {error}"
        )

        # Don't crash the entire graph.
        # Return a basic fallback report.

        result = (
            "# Research Report\n\n"
            "The report-generation agent "
            "was unavailable.\n\n"
            "## Research Summary\n\n"
            f"{state.get('research', '')}\n\n"
            "## Academic Literature\n\n"
            f"{literature_analysis}"
        )

    return {
        "final_report": result
    }


# ============================================================
# RECOMMENDATION NODE
# ============================================================

def recommendation_node(state: ResearchState):

    logger.info("Running Recommendation Agent...")

    topic = state["topic"]

    literature = state.get("literature", {})
    literature_sources = literature.get("sources", [])
    literature_analysis = literature.get("analysis", "")
    
    web_sources = state.get("web_sources", [])
    verified_evidence = state.get("verification_results", [])

    try:

        output = recommendation_agent(
            topic=topic,
            literature_sources=literature_sources,
            web_sources=web_sources,
            literature_analysis=literature_analysis,
            verified_evidence=verified_evidence,
        )

        # Serialise to plain dicts for state storage
        rec_dicts = [
            rec.model_dump()
            for rec in output.recommendations
        ]

        # Append Research Opportunities section to the final report
        existing_report = state.get("final_report", "")

        rec_section = format_recommendations_md(output, topic)

        updated_report = existing_report + rec_section

        logger.info(
            "[Recommendation] Appended %d recommendations to report.",
            len(rec_dicts)
        )

        return {
            "recommendations": rec_dicts,
            "final_report": updated_report,
        }

    except Exception as error:

        logger.error(
            "[Recommendation Node] Failed: %s", error
        )

        return {
            "recommendations": [],
        }


# ============================================================
# BUILD LANGGRAPH
# ============================================================


def _build_graph():
    """Construct and compile the research LangGraph."""

    workflow = StateGraph(ResearchState)

    workflow.add_node("research", research_node)
    workflow.add_node("literature", literature_node)
    workflow.add_node("evidence", evidence_node)
    workflow.add_node("verification", verification_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("report", report_node)
    workflow.add_node("recommendation", recommendation_node)

    workflow.add_edge(START, "research")
    workflow.add_edge("research", "literature")
    workflow.add_edge("literature", "evidence")
    workflow.add_edge("evidence", "verification")
    workflow.add_edge("verification", "critic")

    workflow.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "research": "research",
            "report": "report"
        }
    )

    workflow.add_edge("report", "recommendation")
    workflow.add_edge("recommendation", END)

    return workflow.compile()


# ============================================================
# MAIN FUNCTION
# ============================================================

def run_research(topic):

    logger.info("==============================")
    logger.info("   AI RESEARCH SYSTEM")
    logger.info("==============================")

    # Build the graph fresh for each run
    research_graph = _build_graph()

    # --------------------------------------------------------
    # Initial Graph State
    # --------------------------------------------------------

    initial_state = {

        "topic": topic,

        "research": "",

        "web_sources": [],

        "knowledge_gap": "",

        "follow_up_query": "",

        "literature": {},

        "evidence": [],

        "research_round": 0,

        "new_research_found": False,

        "new_literature_found": False
    }

    # --------------------------------------------------------
    # Run LangGraph
    # --------------------------------------------------------

    final_state = research_graph.invoke(
        initial_state
    )

    # --------------------------------------------------------
    # Return final state
    # --------------------------------------------------------

    return final_state