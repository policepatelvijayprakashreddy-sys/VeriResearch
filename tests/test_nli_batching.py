"""
Unit test for NLI batching and label interpretation.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.verification_agent import (
    _interpret,
    _verify_one,
    _verify_batch,
    _cached_batch,
    ENTAILMENT,
    NEUTRAL,
    CONTRADICTION,
    UNVERIFIED,
)


def test_interpret_correct_labels():
    out = [{"label": "entailment", "score": 0.95}, {"label": "neutral", "score": 0.05}]
    label, score = _interpret(out, threshold=0.70)
    assert label == ENTAILMENT
    assert score == 0.95


def test_interpret_below_threshold_falls_back_to_neutral():
    out = [{"label": "entailment", "score": 0.65}, {"label": "neutral", "score": 0.35}]
    label, score = _interpret(out, threshold=0.70)
    assert label == NEUTRAL
    assert score == 0.65


def test_interpret_label_n_fallback():
    # LABEL_0 -> CONTRADICTION, LABEL_1 -> ENTAILMENT, LABEL_2 -> NEUTRAL
    out0 = [{"label": "LABEL_0", "score": 0.90}]
    out1 = [{"label": "LABEL_1", "score": 0.90}]
    out2 = [{"label": "LABEL_2", "score": 0.90}]
    assert _interpret(out0, 0.70)[0] == CONTRADICTION
    assert _interpret(out1, 0.70)[0] == ENTAILMENT
    assert _interpret(out2, 0.70)[0] == NEUTRAL


def test_interpret_dynamic_id2label():
    id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}
    out = [{"label": "LABEL_1", "score": 0.93}]
    label, score = _interpret(out, 0.70, id2label=id2label)
    assert label == ENTAILMENT
    assert score == 0.93

    out_rev = [{"label": "LABEL_0", "score": 0.88}]
    label_rev, score_rev = _interpret(out_rev, 0.70, id2label=id2label)
    assert label_rev == CONTRADICTION


def test_verify_batch_preserves_order_and_handles_empty():
    def mock_pipe(inputs, *args, **kwargs):
        results = []
        for inp in inputs:
            txt = inp["text"].lower()
            if "supported" in txt:
                results.append([{"label": "ENTAILMENT", "score": 0.95}])
            elif "refuted" in txt:
                results.append([{"label": "CONTRADICTION", "score": 0.92}])
            else:
                results.append([{"label": "NEUTRAL", "score": 0.85}])
        return results

    pairs = [
        ("Claim 1", "This is supported by empirical data."),
        ("", "No claim present."),                         # empty claim -> UNVERIFIED
        ("Claim 3", "This finding is refuted by analysis."),
        ("Claim 4", "   "),                                # whitespace evidence -> UNVERIFIED
        ("Claim 5", "Completely unaddressed matter."),
    ]

    verdicts = _verify_batch(pairs, mock_pipe, threshold=0.70)

    assert len(verdicts) == 5
    assert verdicts[0][0] == ENTAILMENT
    assert verdicts[1][0] == UNVERIFIED
    assert verdicts[2][0] == CONTRADICTION
    assert verdicts[3][0] == UNVERIFIED
    assert verdicts[4][0] == NEUTRAL


def test_cached_batch_hits_disk_cache(tmp_path):
    call_count = 0

    def counting_mock_pipe(inputs, *args, **kwargs):
        nonlocal call_count
        call_count += len(inputs)
        return [[{"label": "ENTAILMENT", "score": 0.95}]] * len(inputs)

    cache_file = str(tmp_path / "test_nli_cache")
    pairs = [
        ("Claim A", "Evidence A text"),
        ("Claim B", "Evidence B text"),
    ]

    # First call: cache miss, invokes pipeline for 2 pairs
    first_res = _cached_batch(pairs, counting_mock_pipe, threshold=0.70, cache_path=cache_file)
    assert call_count == 2
    assert len(first_res) == 2
    assert first_res[0][0] == ENTAILMENT

    # Second call with identical pairs: cache hit, pipeline call count must stay 2!
    second_res = _cached_batch(pairs, counting_mock_pipe, threshold=0.70, cache_path=cache_file)
    assert call_count == 2
    assert second_res == first_res

    # Third call with 1 new pair: only 1 new pair passed to pipeline
    new_pairs = [
        ("Claim A", "Evidence A text"),  # cached
        ("Claim C", "Evidence C text"),  # new
    ]
    third_res = _cached_batch(new_pairs, counting_mock_pipe, threshold=0.70, cache_path=cache_file)
    assert call_count == 3
    assert len(third_res) == 2

