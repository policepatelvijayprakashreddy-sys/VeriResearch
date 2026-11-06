"""
Unit tests for literature_agent.py helpers.

All tests are pure-Python — no network calls, no LLM.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.literature_agent import (
    _classify_source_type,
    _dedup_by_doi,
    _normalize_doi,
    _strip_hallucinated_doi_lines,
    find_invented_dois,
    format_sources,
    analyze_literature,
)


# ============================================================
# DOI normalization
# ============================================================

def test_normalize_doi_strips_url_prefix():
    assert _normalize_doi("https://doi.org/10.1234/abc") == "10.1234/abc"

def test_normalize_doi_strips_doi_prefix():
    assert _normalize_doi("doi:10.1234/abc") == "10.1234/abc"

def test_normalize_doi_strips_trailing_punctuation():
    assert _normalize_doi("10.1234/abc.") == "10.1234/abc"
    assert _normalize_doi("10.1234/abc,") == "10.1234/abc"

def test_normalize_doi_lowercases():
    assert _normalize_doi("10.1234/ABC") == "10.1234/abc"


# ============================================================
# find_invented_dois
# ============================================================

REAL_SOURCE = {"doi": "10.1234/real"}

def test_no_invented_dois_when_all_match():
    analysis = "This paper (10.1234/real) is important."
    assert find_invented_dois(analysis, [REAL_SOURCE]) == set()

def test_detects_invented_doi():
    analysis = "See 10.9999/fake for details."
    invented = find_invented_dois(analysis, [REAL_SOURCE])
    assert "10.9999/fake" in invented

def test_ignores_no_doi_in_analysis():
    analysis = "No DOIs mentioned here."
    assert find_invented_dois(analysis, [REAL_SOURCE]) == set()

def test_allows_doi_with_url_prefix_in_analysis():
    analysis = "Full URL: https://doi.org/10.1234/real was cited."
    assert find_invented_dois(analysis, [REAL_SOURCE]) == set()


# ============================================================
# _strip_hallucinated_doi_lines
# ============================================================

def test_strip_removes_only_bad_line():
    analysis = "Good line.\nBad line citing 10.9999/fake.\nAnother good line."
    result = _strip_hallucinated_doi_lines(analysis, {"10.9999/fake"})
    assert "10.9999/fake" not in result
    assert "Good line." in result
    assert "Another good line." in result

def test_strip_no_op_when_no_invented():
    analysis = "All clean."
    assert _strip_hallucinated_doi_lines(analysis, set()) == "All clean."


# ============================================================
# _classify_source_type
# ============================================================

def test_classify_journal_via_pub_types():
    s = {"publication_types": ["JournalArticle"], "venue": "", "url": ""}
    assert _classify_source_type(s) == "journal"

def test_classify_preprint_via_venue():
    s = {"publication_types": [], "venue": "arXiv", "url": ""}
    assert _classify_source_type(s) == "preprint"

def test_classify_preprint_via_url():
    s = {"publication_types": [], "venue": "", "url": "https://arxiv.org/abs/1234"}
    assert _classify_source_type(s) == "preprint"

def test_classify_conference_via_pub_types():
    s = {"publication_types": ["ConferencePaper"], "venue": "", "url": ""}
    assert _classify_source_type(s) == "conference"

def test_classify_fallback_to_report():
    s = {"publication_types": [], "venue": "", "url": ""}
    assert _classify_source_type(s) == "report"


# ============================================================
# _dedup_by_doi
# ============================================================

def test_dedup_removes_duplicate_doi():
    sources = [
        {"doi": "10.1234/x", "title": "First"},
        {"doi": "10.1234/x", "title": "Duplicate"},
        {"doi": "10.5678/y", "title": "Different"},
    ]
    result = _dedup_by_doi(sources)
    assert len(result) == 2
    assert result[0]["title"] == "First"

def test_dedup_handles_doi_prefix_variants():
    sources = [
        {"doi": "10.1234/x", "title": "Bare DOI"},
        {"doi": "https://doi.org/10.1234/x", "title": "URL DOI"},
    ]
    result = _dedup_by_doi(sources)
    assert len(result) == 1

def test_dedup_keeps_no_doi_sources():
    sources = [
        {"doi": "", "title": "No DOI A"},
        {"doi": "", "title": "No DOI B"},
    ]
    result = _dedup_by_doi(sources)
    assert len(result) == 2


# ============================================================
# format_sources
# ============================================================

def test_format_sources_empty_returns_sentinel():
    assert "NO VERIFIED ACADEMIC SOURCES" in format_sources([])

def test_format_sources_wraps_in_xml():
    s = [{
        "title": "Test Paper", "authors": ["Alice"], "year": 2024,
        "doi": "10.1/t", "url": "https://example.com", "venue": "Nature",
        "content": "Abstract text.", "source": "Semantic Scholar",
        "publication_types": ["JournalArticle"],
    }]
    out = format_sources(s)
    assert '<source id="1">' in out
    assert "</source>" in out
    assert "Abstract text." in out
    assert "Source type: journal" in out

def test_format_sources_caps_abstract():
    long_abstract = "x" * 3000
    s = [{"title": "T", "authors": [], "year": None, "doi": "",
          "url": "", "venue": "", "content": long_abstract, "source": "SS",
          "publication_types": []}]
    out = format_sources(s, abstract_max_chars=1500)
    assert len([line for line in out.splitlines() if "x" in line][0]) <= 1600

def test_format_sources_filters_none_authors():
    s = [{"title": "T", "authors": ["Alice", None, "Bob"], "year": None,
          "doi": "", "url": "", "venue": "", "content": "", "source": "SS",
          "publication_types": []}]
    out = format_sources(s)
    assert "None" not in out
    assert "Alice" in out
    assert "Bob" in out


# ============================================================
# analyze_literature — pure logic (no LLM)
# ============================================================

def test_analyze_literature_returns_sentinel_on_empty_sources():
    result = analyze_literature("any topic", [])
    assert result is not None
    assert "no academic sources" in result.lower()

