import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.evidence_agent import _recover_verbatim_passage


def test_recovers_original_source_sentence():
    source = (
        "The proposed system detects vulnerabilities in "
        "generated code using static analysis."
    )
    candidate = (
        "The proposed system detects vulnerabilities in "
        "generated code using static analysis"
    )
    recovered = _recover_verbatim_passage(candidate, source)
    assert recovered == source


def test_rejects_unrelated_quote():
    source = (
        "The study evaluates generated code on several "
        "security benchmarks."
    )
    candidate = (
        "Quantum computing dramatically improves image "
        "recognition performance across medical datasets."
    )
    assert _recover_verbatim_passage(candidate, source) is None


def test_rejects_empty_quote():
    assert _recover_verbatim_passage("", "Some source text.") is None


def test_recovers_with_smart_quotes_and_unicode_dashes():
    source = (
        "We evaluate cross-language transfer learning across multiple LLM-generated code bases."
    )
    candidate = (
        "“We evaluate cross–language transfer learning across multiple LLM-generated code bases.”"
    )
    recovered = _recover_verbatim_passage(candidate, source)
    assert recovered == source


def test_recovers_multi_sentence_quotation():
    source = (
        "The empirical study benchmarks 15 code generation models. "
        "Our results show a 38% decrease in security vulnerabilities when prompt filters are applied."
    )
    candidate = (
        "The empirical study benchmarks 15 code generation models. "
        "Our results show a 38% decrease in security vulnerabilities when prompt filters are applied"
    )
    recovered = _recover_verbatim_passage(candidate, source)
    assert recovered is not None
    assert "38% decrease in security vulnerabilities" in recovered


def test_recovers_altered_quotation_missing_intro():
    source = (
        "In this work, we propose a hybrid framework combining static analysis with dynamic fuzzing."
    )
    candidate = (
        "propose a hybrid framework combining static analysis with dynamic fuzzing"
    )
    recovered = _recover_verbatim_passage(candidate, source)
    assert recovered == source


def test_rejects_short_quote():
    source = "Static analysis detects vulnerabilities in code."
    candidate = "Static analysis"
    assert _recover_verbatim_passage(candidate, source) is None
