"""
Unit test for literature_node duplicate detection and new_literature_found flag.
"""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orchestrator.orchestrator import literature_node


def test_literature_node_reports_false_when_no_new_papers_in_round_2():
    existing_paper = {
        "title": "Existing Paper",
        "url": "https://doi.org/10.1234/test",
        "doi": "10.1234/test",
        "content": "Paper content about artificial intelligence.",
    }

    state = {
        "topic": "artificial intelligence",
        "research_round": 2,
        "follow_up_query": "ai healthcare",
        "literature": {
            "analysis": "Round 1 analysis",
            "sources": [existing_paper],
        },
    }

    # Mock literature_agent to return the same paper already in state
    mock_lit_result = {
        "analysis": "Updated analysis",
        "sources": [existing_paper],
        "source_count": 1,
    }

    with patch("orchestrator.orchestrator.literature_agent", return_value=mock_lit_result):
        result = literature_node(state)

    # In round 2, because no genuinely new literature was returned,
    # new_literature_found must be False.
    assert result["new_literature_found"] is False


def test_literature_node_reports_true_when_new_paper_found_in_round_2():
    existing_paper = {
        "title": "Existing Paper",
        "url": "https://doi.org/10.1234/test",
        "doi": "10.1234/test",
        "content": "Existing paper content.",
    }
    brand_new_paper = {
        "title": "Brand New Paper",
        "url": "https://doi.org/10.5678/new",
        "doi": "10.5678/new",
        "content": "Brand new paper content.",
    }

    state = {
        "topic": "artificial intelligence",
        "research_round": 2,
        "follow_up_query": "ai healthcare",
        "literature": {
            "analysis": "Round 1 analysis",
            "sources": [existing_paper],
        },
    }

    mock_lit_result = {
        "analysis": "Updated analysis with new paper",
        "sources": [brand_new_paper],
        "source_count": 1,
    }

    with patch("orchestrator.orchestrator.literature_agent", return_value=mock_lit_result):
        result = literature_node(state)

    assert result["new_literature_found"] is True

