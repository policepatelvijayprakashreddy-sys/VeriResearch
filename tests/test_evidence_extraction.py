import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.evidence_agent import (
    _recover_verbatim_evidence,
    _stable_source_key,
    _stable_evidence_id,
)


def test_verbatim_exact_match():
    content = "This is a study on neural code generation models and empirical security flaws."
    raw = "neural code generation models and empirical security flaws"
    recovered = _recover_verbatim_evidence(raw, content)
    assert raw in recovered


def test_harmless_formatting_quotes_and_whitespace():
    content = 'The authors state: "Static analysis alone fails to uncover 40% of runtime vulnerabilities."'
    raw = '“Static analysis alone fails to uncover 40% of runtime vulnerabilities.”'
    recovered = _recover_verbatim_evidence(raw, content)
    assert recovered is not None
    assert "40% of runtime vulnerabilities" in recovered


def test_recover_sentence_with_minor_dropped_intro():
    content = (
        "We propose a hybrid framework combining static analysis with Bandit and dynamic fuzzing with AFL "
        "to detect and mitigate vulnerabilities in LLM-generated code without retraining. "
        "Experimental evaluations show significant security improvements."
    )
    raw = "proposes a hybrid framework combining static analysis with Bandit and dynamic fuzzing with AFL to detect and mitigate vulnerabilities"
    recovered = _recover_verbatim_evidence(raw, content)
    assert recovered is not None
    assert "We propose a hybrid framework" in recovered


def test_unicode_dash_normalization():
    content = "Cross-language transfer learning improves vulnerability detection across programming languages."
    raw = "Cross–language transfer learning improves vulnerability detection"  # en-dash
    recovered = _recover_verbatim_evidence(raw, content)
    assert recovered is not None
    assert "Cross-language transfer learning" in recovered


def test_fabricated_quote_rejection():
    content = "This paper focuses on static analysis of Java and C++ code."
    fabricated = "Our deep learning agent automatically fixes 99% of memory leaks with zero human oversight."
    recovered = _recover_verbatim_evidence(fabricated, content)
    assert recovered is None


def test_too_short_quote_rejection():
    content = "This paper focuses on static analysis of Java and C++ code."
    short_quote = "This paper"
    recovered = _recover_verbatim_evidence(short_quote, content)
    assert recovered is None


def test_stable_evidence_id_deterministic():
    src_key = "src_12345678"
    ev = "Verbatim evidence snippet"
    claim = "This is an empirical claim"
    id1 = _stable_evidence_id(src_key, ev, claim)
    id2 = _stable_evidence_id(src_key, ev, claim)
    assert id1 == id2
    assert id1.startswith("ev_")
