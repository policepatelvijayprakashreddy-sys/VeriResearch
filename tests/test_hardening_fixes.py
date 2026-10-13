import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.eval_harness import _citation_coverage
from search.source_quality_manager import verify_single_paper
from orchestrator.orchestrator import _build_graph, filter_relevant_sources


def test_citation_coverage_direct_report_text():
    """Verify _citation_coverage accepts report text directly and computes sentence-level coverage."""
    report_text = (
        "## 1. Executive Summary\n\n"
        "This is an empirical study investigating vulnerability detection [1].\n"
        "Here is an uncited factual statement about machine learning systems."
    )
    # 1 cited sentence out of 2 eligible = 0.50 (50%)
    cov = _citation_coverage(report_text, source_count=5)
    assert cov == 0.5


def test_source_quality_ebook_venue_false_positive():
    """
    Verify that when title similarity >= 0.80 and authors match for dual-registry verified paper,
    an eBook platform venue difference alone does NOT force identity_conflict.
    """
    mock_paper = {
        "title": "On deterring hate speech, while maximizing security and privacy",
        "doi": "10.1016/b978-0-443-18851-0.00021-4",
        "venue": "Ethics in Online AI-based Systems",
        "url": "https://doi.org/10.1016/b978-0-443-18851-0.00021-4",
        "authors": ["Sue Spaid"],
    }
    profile = verify_single_paper(mock_paper)

    # Title similarity should be 1.00 and authors should match
    assert profile["title_similarity"] >= 0.80
    assert profile["authors_match"] is True
    # Crucial assertion: eBook platform venue disparity must NOT trigger identity_conflict
    assert profile["identity_conflict"] is False
    assert profile["identity_status"] != "IDENTITY_MISMATCH"


def test_orchestrator_recommendation_edge():
    """Verify that LangGraph workflow routes report -> recommendation -> END."""
    graph = _build_graph()
    edges = graph.builder.edges

    # Check that ('report', 'recommendation') and ('recommendation', '__end__') exist
    edge_pairs = [(u, v) for u, v in edges]
    assert ("report", "recommendation") in edge_pairs
    assert ("recommendation", "__end__") in edge_pairs


def test_relevance_filter_drops_off_topic_sources():
    """Verify filter_relevant_sources drops off-topic sources and keeps on-topic ones."""
    topic = "Security vulnerabilities in LLM-generated code"

    sources = [
        {
            "title": "Evaluating Security and Inclusivity in LLM-Generated Code",
            "content": "This paper analyzes vulnerabilities in code produced by LLMs.",
        },
        {
            "title": "Meeting Compliance Requirements While Using Cloud Services",
            "content": "Discusses HIPAA, Sarbanes-Oxley, and FISMA compliance for enterprise storage.",
        },
    ]

    filtered = filter_relevant_sources(sources, topic, min_keyword_matches=2)
    assert len(filtered) == 1
    assert "LLM-Generated Code" in filtered[0]["title"]


def test_relevance_filter_safe_fallback():
    """Verify filter_relevant_sources safely falls back to original list if all are flagged."""
    topic = "Quantum teleportation algorithms"
    sources = [
        {"title": "Paper Alpha", "content": "Unrelated content A"},
        {"title": "Paper Beta", "content": "Unrelated content B"},
    ]
    # When no sources match, filter returns original list rather than empty list
    filtered = filter_relevant_sources(sources, topic)
    assert len(filtered) == 2
