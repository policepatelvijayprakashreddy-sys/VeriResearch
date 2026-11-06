"""
Unit tests for BM25 Okapi sentence-window retrieval and consensus ranking.
"""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agents.consensus_agent as consensus_module

from agents.consensus_agent import (
    _windows,
    _find_best_passage_bm25,
    BM25Okapi,
)


def test_windows_various_sentence_counts():
    # Empty
    assert _windows("") == []
    assert _windows("   ") == []

    # Single sentence
    s1 = "NumPy provides vectorization for array programming."
    assert _windows(s1, size=2) == [s1]

    # Two sentences
    s2 = "First sentence. Second sentence."
    assert _windows(s2, size=2) == ["First sentence. Second sentence."]

    # Three sentences -> 2 sliding windows
    s3 = "Sentence one. Sentence two. Sentence three."
    w3 = _windows(s3, size=2)
    assert len(w3) == 2
    assert w3[0] == "Sentence one. Sentence two."
    assert w3[1] == "Sentence two. Sentence three."

    # Four sentences -> 3 sliding windows
    s4 = "Sentence 1. Sentence 2. Sentence 3. Sentence 4."
    w4 = _windows(s4, size=2)
    assert len(w4) == 3
    assert w4[0] == "Sentence 1. Sentence 2."
    assert w4[1] == "Sentence 2. Sentence 3."
    assert w4[2] == "Sentence 3. Sentence 4."


def test_bm25_okapi_ranking():
    corpus = [
        ["deep", "learning", "convolutional", "networks", "image", "classification"],
        ["historical", "architecture", "roman", "aqueducts", "masonry", "mortar"],
        ["vectorized", "array", "programming", "numpy", "speedups", "performance"],
    ]
    bm25 = BM25Okapi(corpus)

    query = ["numpy", "vectorized", "performance"]
    scores = bm25.get_scores(query)

    assert len(scores) == 3
    # Document 2 (index 2) must have highest score
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    assert best_idx == 2
    assert scores[2] > scores[0]
    assert scores[1] == 0.0  # Completely unrelated


def test_find_best_passage_bm25_retrieves_focused_window():
    abstract = (
        "In this study we examine computational bottlenecks across diverse environments. "
        "Prior approaches relied on standard interpreted loops. "
        "Vectorized array calculations using NumPy libraries deliver orders-of-magnitude speedups over pure Python loops. "
        "Furthermore memory overhead is reduced by contiguous buffer allocation. "
        "Future work will explore distributed GPU clusters."
    )
    claim = "NumPy array programming achieves significant speedups through vectorization."

    passage, score = _find_best_passage_bm25(claim, abstract, max_chars=400, size=2)

    assert score > 0.0
    assert "Vectorized array calculations using NumPy" in passage
    assert "speedups" in passage
    # The irrelevant opening sentence should not be the primary premise
    assert "In this study we examine computational bottlenecks" not in passage


def test_find_best_passage_bm25_zero_score_on_unrelated_content():
    roman_aqueducts = (
        "Analysis of ancient hydraulic mortar reveals sophisticated masonry techniques utilized in the Mediterranean basin. "
        "Vitruvian guidelines influenced brick and mortar construction across imperial provinces. "
        "Excavations demonstrate enduring architectural stability."
    )
    claim = "Cryogenic quantum-dot laser diodes achieve 99.8% modulation efficiency at 4 Kelvin."

    passage, score = _find_best_passage_bm25(claim, roman_aqueducts, max_chars=400, size=2)

    assert score == 0.0


def test_windows_with_academic_abbreviations():
    text = (
        "Several methods were evaluated, e.g. convolutional networks. "
        "The results showed improved classification accuracy. "
        "Fig. 2 presents the experimental comparison."
    )
    windows = _windows(text, size=2)
    assert len(windows) == 2
    assert "e.g. convolutional networks." in windows[0]
    assert "Fig. 2 presents" in windows[1]


def test_windows_with_decimal_numbers():
    text = (
        "The model achieved 98.7% accuracy. "
        "The baseline achieved 94.2% accuracy. "
        "The proposed approach performed better."
    )
    windows = _windows(text, size=2)
    assert len(windows) == 2
    assert "98.7%" in windows[0]
    assert "94.2%" in windows[0]


def test_bm25_fallback_rejects_unrelated_content(monkeypatch):
    def failing_bm25(*args, **kwargs):
        raise RuntimeError("Simulated BM25 failure")

    monkeypatch.setattr(
        consensus_module,
        "BM25Okapi",
        failing_bm25,
    )
    claim = "Quantum dot lasers improve optical modulation efficiency."
    content = (
        "Ancient Roman aqueducts used sophisticated masonry "
        "and hydraulic mortar construction."
    )
    passage, score = consensus_module._find_best_passage_bm25(
        claim,
        content,
    )
    assert passage == ""
    assert score == 0.0


def test_bm25_fallback_accepts_relevant_content(monkeypatch):
    def failing_bm25(*args, **kwargs):
        raise RuntimeError("Simulated BM25 failure")

    monkeypatch.setattr(
        consensus_module,
        "BM25Okapi",
        failing_bm25,
    )
    claim = "NumPy vectorization improves numerical computing performance."
    content = (
        "Vectorized calculations using NumPy improve numerical computing performance "
        "compared to standard interpreted loops."
    )
    passage, score = consensus_module._find_best_passage_bm25(
        claim,
        content,
    )
    assert passage != ""
    assert score > 0.0


def test_lexical_prefilter_semantic_paraphrase():
    claim = "Deep learning improves image classification accuracy."
    passage = (
        "Neural network architectures achieve higher recognition rates."
    )
    keywords = consensus_module._extract_claim_keywords(claim)
    # Diagnostic: demonstrate the current lexical limitation.
    assert not consensus_module._lexical_prefilter(
        keywords,
        passage,
        min_overlap=2,
    )


def test_semantic_retrieval_recovers_paraphrase(monkeypatch):
    claim = "Deep learning improves image classification accuracy."
    passage = "Neural network architectures achieve higher recognition rates."

    def mock_encode(text, model_name):
        if text == claim:
            return (1.0, 0.0)
        return (0.508, 0.862)

    monkeypatch.setattr(
        consensus_module,
        "_encode_semantic_text",
        mock_encode,
    )

    result, score = consensus_module._find_best_passage_semantic(
        claim, passage
    )

    assert result == passage
    assert score >= 0.40


def test_semantic_retrieval_rejects_unrelated_content(monkeypatch):
    claim = "Deep learning improves image classification accuracy."
    passage = "Ancient Roman aqueducts used sophisticated masonry."

    def mock_encode(text, model_name):
        if text == claim:
            return (1.0, 0.0)
        return (0.0, 1.0)

    monkeypatch.setattr(
        consensus_module,
        "_encode_semantic_text",
        mock_encode,
    )

    result, score = consensus_module._find_best_passage_semantic(
        claim, passage
    )

    assert result == ""
    assert score < 0.40


def test_semantic_retrieval_also_recovers_contradictions(monkeypatch):
    claim = "Deep learning improves image classification accuracy."
    passage = "Deep learning does not improve image classification accuracy."

    def mock_encode(text, model_name):
        if text == claim:
            return (1.0, 0.0)
        return (0.761, 0.649)

    monkeypatch.setattr(
        consensus_module,
        "_encode_semantic_text",
        mock_encode,
    )

    result, score = consensus_module._find_best_passage_semantic(
        claim, passage
    )

    assert result == passage
    assert score >= 0.40


def test_candidate_retrieval_prefers_bm25(monkeypatch):
    claim = "NumPy vectorization improves numerical computing performance."
    content = "NumPy vectorization improves numerical computing performance."

    monkeypatch.setattr(
        consensus_module,
        "_lexical_prefilter",
        lambda *args, **kwargs: True,
    )

    monkeypatch.setattr(
        consensus_module,
        "_find_best_passage_bm25",
        lambda *args, **kwargs: (content, 2.5),
    )

    def unexpected_semantic_call(*args, **kwargs):
        raise AssertionError("Semantic fallback should not run")

    monkeypatch.setattr(
        consensus_module,
        "_find_best_passage_semantic",
        unexpected_semantic_call,
    )

    passage, score, method = (
        consensus_module._retrieve_candidate_passage(
            claim,
            content,
            consensus_module._extract_claim_keywords(claim),
        )
    )

    assert passage == content
    assert score == 2.5
    assert method == "bm25"





