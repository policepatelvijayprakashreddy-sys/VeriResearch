import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.report_agent import (
    _normalize_doi,
    _source_aliases,
    _audit_citations,
)


def test_doi_normalization():
    assert _normalize_doi(
        "https://doi.org/10.1234/ABC"
    ) == "10.1234/abc"


def test_doi_alias_matches_resolver_url():
    source_a = {"doi": "10.1234/ABC"}
    source_b = {"url": "https://doi.org/10.1234/abc"}
    assert set(_source_aliases(source_a)) & set(
        _source_aliases(source_b)
    )


def test_valid_and_invalid_citation_markers():
    body = (
        "The system improved performance across the evaluated tasks [1].\n"
        "The results establish a second independent conclusion [9]."
    )
    metrics = _audit_citations(body, source_count=2)
    assert metrics["total_marker_count"] == 2
    assert metrics["invalid_marker_count"] == 1
    assert metrics["eligible_claims"] == 2
    assert metrics["cited_claims"] == 1


if __name__ == "__main__":
    test_doi_normalization()
    test_doi_alias_matches_resolver_url()
    test_valid_and_invalid_citation_markers()
    print("All test_report_citations passed [OK]")
