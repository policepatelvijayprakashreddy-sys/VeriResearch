import logging
import warnings
from typing import TypedDict, Any

# Suppress LangChain / LangGraph internal deprecation notices
try:
    from langchain_core._api.deprecation import (
        LangChainDeprecationWarning,
        LangChainPendingDeprecationWarning,
    )
    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
    warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)
except ImportError:
    pass
warnings.filterwarnings("ignore")

from langgraph.graph import StateGraph, START, END

from agents.research_agent import research_agent
from agents.literature_agent import literature_agent
from agents.source_quality_agent import source_quality_agent
from agents.evidence_agent import extract_evidence
from agents.critic_agent import critic_agent
from agents.report_agent import report_agent
from agents.recommendation_agent import (
    recommendation_agent,
    format_recommendations_md,
)
from agents.verification_agent import verification_agent
from agents.consensus_agent import consensus_agent

import config
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

    # Source quality & verification
    source_quality_profiles: list
    source_quality_table: str
    retraction_warnings: list

    # Extracted evidence
    evidence: list
    verification_results: list
    hallucination_rate: float

    # Cross-paper consensus
    consensus_profiles: list
    consensus_table: str

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
        follow_up = state.get("follow_up_query", "") if current_round > 1 else ""
        result = research_agent(
            state["topic"],
            previous_research,
            follow_up_query=follow_up
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

    prev_web_urls = {s.get("url", "").strip().lower() for s in previous_web_sources if s.get("url")}
    truly_new_web = [s for s in new_sources if s.get("url", "").strip().lower() not in prev_web_urls]
    new_sources_found = bool(truly_new_web) if current_round > 1 else bool(new_sources)

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

    follow_up = state.get("follow_up_query", "") if state.get("research_round", 1) > 1 else ""

    try:

        result = literature_agent(
            state["topic"],
            follow_up_query=follow_up
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

    prev_lit_urls = {s.get("url", "").strip().lower() for s in previous_sources if s.get("url")}
    truly_new_lit = [s for s in new_sources if s.get("url", "").strip().lower() not in prev_lit_urls]
    new_literature_found = bool(truly_new_lit) if state.get("research_round", 1) > 1 else bool(new_sources)

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
    # Relevance Pre-Filter (Drops off-topic keyword false-positives)
    # --------------------------------------------------------
    import re
    STOP_FILTER = {"a", "an", "the", "in", "on", "of", "and", "for", "with", "to", "at", "by", "from", "is", "are", "via", "using", "as", "system", "study", "paper"}
    topic_keywords = {w for w in re.findall(r"[a-z0-9]{3,}", state.get("topic", "").lower()) if w not in STOP_FILTER}

    if topic_keywords:
        relevant_sources = []
        for s in unique_sources:
            text = f"{s.get('title', '')} {s.get('content', '')}".lower()
            match_count = sum(1 for w in topic_keywords if w in text)
            if match_count >= min(2, len(topic_keywords)):
                relevant_sources.append(s)
        if relevant_sources:
            logger.info("[Literature Node] Relevance filter kept %d/%d academic papers.", len(relevant_sources), len(unique_sources))
            unique_sources = relevant_sources

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
# SOURCE QUALITY NODE
# ============================================================

def source_quality_node(state: ResearchState):
    """
    Verify paper provenance, retraction status, DOI identity,
    and journal registration for all gathered academic sources.
    """
    logger.info("Running Source Quality Verification Node...")

    literature = state.get("literature", {})
    literature_sources = literature.get("sources", [])

    if not literature_sources:
        return {
            "source_quality_profiles": [],
            "source_quality_table": "",
            "retraction_warnings": []
        }

    audit = source_quality_agent(literature_sources)

    # Keep literature sources enriched with verified quality fields
    literature["sources"] = audit["enriched_sources"]

    return {
        "literature": literature,
        "source_quality_profiles": audit["profiles"],
        "source_quality_table": audit["markdown_table"],
        "retraction_warnings": audit["retraction_warnings"]
    }


def _norm_text(s: str) -> str:
    import re
    return re.sub(r"\W+", " ", (s or "").lower()).strip()


def evidence_in_source(evidence: str, source: dict, min_ratio: float = 0.70) -> bool:
    import difflib
    ev = _norm_text(evidence)
    src = _norm_text(f"{source.get('title','')} {source.get('content','')}")
    if not ev or not src:
        return False
    if ev in src:
        return True
    m = difflib.SequenceMatcher(None, src, ev, autojunk=False).find_longest_match(0, len(src), 0, len(ev))
    return (m.size / len(ev)) >= min_ratio


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
            claim_text = (getattr(item, "claim", "") or "").strip()
            evidence_text = (getattr(item, "evidence", "") or "").strip()
            if not claim_text or not evidence_text or len(claim_text) < 5:
                continue

            source_id = item.source_id.strip()
            source = source_records.get(source_id)

            if source is None and f"source_{source_id}" in source_records:
                source_id = f"source_{source_id}"
                source = source_records[source_id]

            if source is None:
                item_url = getattr(item, "source_url", "").strip().lower()
                matched_source = (
                    sources_by_url.get(source_id.lower())
                    or (sources_by_url.get(item_url) if item_url else None)
                )
                if matched_source:
                    source_id, source = matched_source

            if source is None:
                continue

            # Verify that the extracted evidence actually appears in the source content
            if not evidence_in_source(evidence_text, source):
                logger.debug(
                    "[Evidence Node] Dropping mismatched quote for %s: %s",
                    source_id, evidence_text[:60]
                )
                continue

            evidence_items.append({
                "source_id": source_id,
                "source_title": source.get(
                    "title",
                    item.source_title
                ),
                "source_url": source.get("url", ""),
                "evidence": evidence_text,
                "claim": claim_text,
                "venue": source.get("journal_name", source.get("venue", "")),
                "issn": source.get("issn", ""),
                "doi": source.get("doi", ""),
                "is_retracted": source.get("is_retracted", False),
                "verified_indexes": source.get("verified_indexes", []),
                "confidence_level": source.get("confidence_level", "LOW"),
                "harvested_from": source.get("source", "Web Search"),
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
# CONSENSUS NODE (Stage 2: Cross-Paper Claim Consensus)
# ============================================================

def consensus_node(state: ResearchState):
    logger.info("Running Cross-Paper Consensus Agent...")
    evidence_items = state.get("evidence", [])
    verification_results = state.get("verification_results", [])
    literature = state.get("literature", {})
    literature_sources = literature.get("sources", [])

    if not evidence_items or not literature_sources:
        return {
            "consensus_profiles": [],
            "consensus_table": "",
        }

    try:
        res = consensus_agent(
            evidence_items=evidence_items,
            verification_results=verification_results,
            literature_sources=literature_sources,
            max_claims=5,
            topic=state.get("topic", ""),
        )
        return {
            "consensus_profiles": res.get("profiles", []),
            "consensus_table": res.get("markdown_table", ""),
        }
    except Exception as error:
        logger.error("[Consensus Node] Failed: %s", error)
        return {
            "consensus_profiles": [],
            "consensus_table": "",
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
            ),
            source_quality_table=state.get("source_quality_table", ""),
            retraction_warnings=state.get("retraction_warnings", []),
            consensus_table=state.get("consensus_table", ""),
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

    knowledge_gap = getattr(result, "knowledge_gap", "")
    follow_up_query = getattr(result, "follow_up_query", "")
    if follow_up_query:
        topic_words = state.get("topic", "").split()[:4]
        topic_prefix = " ".join(topic_words)
        if topic_prefix.lower() not in follow_up_query.lower():
            follow_up_query = f"{topic_prefix} {follow_up_query}"

    return {
        "critique": result,
        "decision": decision,
        "knowledge_gap": knowledge_gap,
        "follow_up_query": follow_up_query,
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
        and research_round >= MAX_RESEARCH_ROUNDS
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
    verification_results = state.get("verification_results", [])

    # If verification is enabled, strictly filter out contradicted (hallucinated) claims
    if config.ENABLE_VERIFICATION and verification_results:
        verified_map = {r.get("claim", "").strip(): r.get("label") for r in verification_results}
        uncontradicted_evidence = [
            ev for ev in evidence
            if verified_map.get(ev.get("claim", "").strip()) != "CONTRADICTION"
        ]
        report_evidence = uncontradicted_evidence if uncontradicted_evidence else evidence
        logger.info(
            "[Report Node] Evidence count: %d total, %d passed verification (dropped %d contradicted).",
            len(evidence), len(report_evidence), len(evidence) - len(report_evidence)
        )
    else:
        report_evidence = evidence
        logger.info("[Report Node] Verification disabled; passing all %d evidence items.", len(evidence))

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

            report_evidence,
            
            verification_results,
            source_quality_table=state.get("source_quality_table", ""),
            consensus_table=state.get("consensus_table", ""),
            source_quality_profiles=state.get("source_quality_profiles", []),
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
    workflow.add_node("source_quality", source_quality_node)
    workflow.add_node("evidence", evidence_node)
    workflow.add_node("verification", verification_node)
    workflow.add_node("consensus", consensus_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("report", report_node)
    workflow.add_node("recommendation", recommendation_node)

    workflow.add_edge(START, "research")
    workflow.add_edge("research", "literature")
    workflow.add_edge("literature", "source_quality")
    workflow.add_edge("source_quality", "evidence")
    workflow.add_edge("evidence", "verification")
    workflow.add_edge("verification", "consensus")
    workflow.add_edge("consensus", "critic")

    workflow.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "research": "research",
            "report": "report"
        }
    )

    workflow.add_edge("report", END)

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

        "source_quality_profiles": [],

        "source_quality_table": "",

        "retraction_warnings": [],

        "evidence": [],

        "consensus_profiles": [],

        "consensus_table": "",

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