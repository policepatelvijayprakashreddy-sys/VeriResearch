import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.eval_harness import (
    _citation_coverage,
    _extract_citation_audit_counts,
    run_eval,
)


def test_citation_coverage_zero_percent():
    """Verify 0% citation coverage when eligible claims exist but none are cited."""
    assert _citation_coverage(0, 10) == 0.0


def test_citation_coverage_partial():
    """Verify partial citation coverage (e.g. 50% and 33.33%)."""
    assert _citation_coverage(3, 6) == 0.5
    assert round(_citation_coverage(1, 3), 4) == 0.3333


def test_citation_coverage_one_hundred_percent():
    """Verify 100% citation coverage when all eligible claims are cited."""
    assert _citation_coverage(5, 5) == 1.0


def test_citation_coverage_zero_eligible_candidates():
    """Verify safe zero division handling when zero eligible candidates exist."""
    assert _citation_coverage(0, 0) == 0.0
    assert _citation_coverage(3, 0) == 0.0
    assert _citation_coverage(0, -1) == 0.0


def test_extract_counts_from_audit_section():
    """Verify extraction of counts when ## Citation Integrity Audit is present."""
    report_text = """
## 1. Executive Summary
This is a test report.

## Citation Integrity Audit

- Eligible factual-claim candidates: 12
- Candidates with at least one registered citation: 6
- Claim-level citation coverage: 50.00%
- Invalid citation marker occurrences: 0
- Total inline citation marker occurrences: 6

### Sources
[1] Source A
"""
    cited, eligible = _extract_citation_audit_counts(report_text, source_count=1)
    assert eligible == 12
    assert cited == 6
    assert _citation_coverage(cited, eligible) == 0.5


def test_extract_counts_fallback():
    """Verify fallback when report lacks ## Citation Integrity Audit section."""
    raw_report = (
        "## 1. Executive Summary\n\n"
        "The model demonstrates state of the art performance on benchmarks [1].\n"
        "Another sentence without any citation marker is presented here."
    )
    cited, eligible = _extract_citation_audit_counts(raw_report, source_count=1)
    assert eligible == 2
    assert cited == 1
    assert _citation_coverage(cited, eligible) == 0.5


def test_run_eval_citation_coverage_integration():
    """Verify run_eval uses deterministic claim-level coverage."""
    report = (
        "# Research Report: Test Topic\n\n"
        "## 1. Executive Summary\nTest summary with valid claims [1].\n\n"
        "## 2. General Research Findings\nGeneral research without citations.\n\n"
        "## 3. Literature Review\nLiterature findings without citations.\n\n"
        "## 4. Critical Analysis\nAnalysis content without citations.\n\n"
        "## 5. Knowledge Gaps\n" + ("Extensive knowledge gap discussion text word count " * 10) + "\n\n"
        "## 6. Recommendations\nSpecific recommendations without citations.\n\n"
        "## 7. Conclusion\nConclusion statements without citations.\n\n"
        "## Citation Integrity Audit\n\n"
        "- Eligible factual-claim candidates: 4\n"
        "- Candidates with at least one registered citation: 1\n"
        "- Claim-level citation coverage: 25.00%\n"
        "- Invalid citation marker occurrences: 0\n"
        "- Total inline citation marker occurrences: 1\n\n"
        "### Sources\n\n[1] Source One"
    )
    state = {
        "topic": "Test Topic",
        "final_report": report,
        "web_sources": [{"url": "http://example.com/1"}],
        "literature": {"sources": []},
        "evidence": [{"claim": "c1", "evidence": "e1"}, {"claim": "c2", "evidence": "e2"}],
    }
    eval_rep = run_eval(state, use_llm_judge=False)
    # 1 cited out of 4 eligible = 0.25 (25%), NOT 1 citation / 2 evidence = 0.50
    assert eval_rep.citation_coverage == 0.25
