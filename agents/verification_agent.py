"""
Verification Agent (NLI-based)
==============================

Objectively verifies whether extracted research claims are actually
supported by the source text they are attributed to.

For each EvidenceItem extracted by the Evidence Agent:
    premise    = item["evidence"]  (the raw snippet fetched from the source)
    hypothesis = item["claim"]     (the claim the LLM derived from that text)

The NLI model classifies each pair as:
    ENTAILMENT    — source text entails / supports the claim        (trustworthy)
    NEUTRAL       — source text is unrelated or insufficient        (suspicious)
    CONTRADICTION — source text directly contradicts the claim      (hallucination)

Metrics produced:
    entailment_rate    = ENTAILMENT  / total
    neutral_rate       = NEUTRAL     / total
    contradiction_rate = CONTRADICTION / total
    hallucination_rate = (NEUTRAL + CONTRADICTION) / total   ← headline metric

Model: cross-encoder/nli-deberta-v3-small (~185 MB, CPU-compatible)
       Downloaded once to ~/.cache/huggingface/ on first run.

Graceful fallback:
    If transformers / torch is not installed, every item gets label=UNVERIFIED.
    The pipeline continues without crashing; hallucination_rate is set to None.
"""

import hashlib
import logging
import os
import shelve
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path when running script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from transformers import pipeline as hf_pipeline
except ImportError:
    hf_pipeline = None

import config

logger = logging.getLogger(__name__)

# NLI label constants
ENTAILMENT    = "ENTAILMENT"
NEUTRAL       = "NEUTRAL"
CONTRADICTION = "CONTRADICTION"
UNVERIFIED    = "UNVERIFIED"   # fallback when NLI is unavailable


# ============================================================
# DATA MODELS (plain dicts — no pydantic dep needed here)
# ============================================================

# VerificationResult keys:
#   claim           : str   — the derived claim
#   evidence        : str   — the raw source snippet
#   source_url      : str   — where the claim came from
#   source_title    : str
#   label           : str   — ENTAILMENT | NEUTRAL | CONTRADICTION | UNVERIFIED
#   confidence      : float | None
#   citation_number : int | None  — filled in later by report_agent

# VerificationSummary keys:
#   results            : list[VerificationResult]
#   entailment_count   : int
#   neutral_count      : int
#   contradiction_count: int
#   unverified_count   : int
#   total              : int
#   entailment_rate    : float | None
#   neutral_rate       : float | None
#   contradiction_rate : float | None
#   hallucination_rate : float | None   — primary metric


# ============================================================
# NLI MODEL — LAZY SINGLETON
# ============================================================

_nli_pipeline = None


def _load_nli_pipeline():
    """
    Load the NLI pipeline once and cache it in _nli_pipeline.
    Returns None if transformers/torch is not installed.
    """
    global _nli_pipeline

    if _nli_pipeline is not None:
        return _nli_pipeline

    if hf_pipeline is None:
        logger.warning(
            "[Verification] transformers / torch not installed. "
            "All claims will be labelled UNVERIFIED."
        )
        return None

    try:
        logger.info(
            "[Verification] Loading NLI model: %s (CPU text-classification)...",
            config.NLI_MODEL_NAME
        )

        _nli_pipeline = hf_pipeline(
            "text-classification",
            model=config.NLI_MODEL_NAME,
            device=-1,           # CPU
        )
        logger.info("[Verification] NLI labels: %s", _nli_pipeline.model.config.id2label)
        return _nli_pipeline

    except Exception as error:
        logger.error(
            "[Verification] Failed to load NLI model: %s. "
            "All claims will be labelled UNVERIFIED.", error
        )
        return None


def _make_verification_result(
    item: dict,
    label: str,
    confidence: Optional[float],
) -> dict:
    """Preserve evidence provenance through verification."""
    return {
        "source_id": item.get("source_id", ""),
        "source_key": item.get("source_key", ""),
        "evidence_id": item.get("evidence_id", ""),
        "claim": item.get("claim", ""),
        "evidence": item.get("evidence", ""),
        "source_url": item.get("source_url", ""),
        "source_title": item.get("source_title", ""),
        "label": label,
        "confidence": (
            round(confidence, 3)
            if confidence is not None
            else None
        ),
    }


# ============================================================
# VERIFY A SINGLE CLAIM OR BATCH
# ============================================================

def _interpret(
    out,
    threshold: float,
    id2label: Optional[dict] = None,
) -> tuple[str, Optional[float]]:
    """Convert raw pipeline scores for one pair into (label, score)."""
    if out and isinstance(out, dict):
        out = [out]
    best = max(out, key=lambda o: o["score"])
    raw_label = best["label"]
    label = str(raw_label).upper()

    # Dynamic id2label resolution from model config if available
    if id2label:
        mapped = None
        if raw_label in id2label:
            mapped = id2label[raw_label]
        elif str(raw_label).isdigit() and int(raw_label) in id2label:
            mapped = id2label[int(raw_label)]
        elif label.startswith("LABEL_"):
            idx_str = label.split("_")[-1]
            if idx_str.isdigit():
                mapped = id2label.get(int(idx_str)) or id2label.get(idx_str)
        if mapped:
            label = str(mapped).upper()

    if label not in (ENTAILMENT, NEUTRAL, CONTRADICTION):
        # cross-encoder/nli-deberta-v3-small id2label:
        # 0: contradiction, 1: entailment, 2: neutral
        label_map = {
            "LABEL_0": CONTRADICTION,
            "LABEL_1": ENTAILMENT,
            "LABEL_2": NEUTRAL,
        }
        label = label_map.get(label, NEUTRAL)

    if label != NEUTRAL and best["score"] < threshold:
        return NEUTRAL, best["score"]
    return label, best["score"]


def _verify_one(
    claim: str,
    evidence: str,
    pipe,
    threshold: float,
) -> tuple[str, Optional[float]]:
    """
    Run NLI on a single (premise=evidence, hypothesis=claim) pair.

    Returns
    -------
    (label, confidence)
        label      : ENTAILMENT | NEUTRAL | CONTRADICTION | UNVERIFIED
        confidence : float 0.0-1.0 for the winning label
    """
    if not evidence.strip() or not claim.strip():
        return UNVERIFIED, None

    try:
        model_cfg = getattr(getattr(pipe, "model", None), "config", None)
        id2label = getattr(model_cfg, "id2label", None) if model_cfg is not None else None
        out = pipe(
            {"text": evidence[:1024], "text_pair": claim[:256]},
            top_k=None,
            truncation=True,
        )
        if out and isinstance(out[0], list):
            out = out[0]
        return _interpret(out, threshold, id2label=id2label)
    except Exception as e:
        logger.debug("[Verification] NLI failed: %s", e)
        return NEUTRAL, None


def _verify_batch(
    pairs: list[tuple[str, str]],   # [(claim, evidence), ...]
    pipe,
    threshold: float,
    batch_size: int = 16,
) -> list[tuple[str, Optional[float]]]:
    """
    Batched equivalent of calling _verify_one on every pair, preserving original input order.
    """
    results: list[tuple[str, Optional[float]]] = [(UNVERIFIED, None)] * len(pairs)

    # Only run valid pairs; sort by length so batches have minimal padding
    valid = [
        i for i, (c, e) in enumerate(pairs)
        if (c or "").strip() and (e or "").strip()
    ]
    if not valid:
        return results

    valid.sort(key=lambda i: len((pairs[i][1] or "")[:1024]) + len((pairs[i][0] or "")[:256]))

    inputs = [
        {"text": pairs[i][1][:1024], "text_pair": pairs[i][0][:256]}
        for i in valid
    ]

    model_cfg = getattr(getattr(pipe, "model", None), "config", None)
    id2label = getattr(model_cfg, "id2label", None) if model_cfg is not None else None

    try:
        outs = pipe(inputs, batch_size=batch_size, top_k=None, truncation=True)
        # Ensure outs matches valid batch format (list of lists or list of dicts)
        if isinstance(outs, list) and outs and isinstance(outs[0], list) and len(outs) == len(valid):
            for i, out in zip(valid, outs):
                results[i] = _interpret(out, threshold, id2label=id2label)
        elif len(valid) == 1 and isinstance(outs, list):
            # Single-pair call might return single list of label dicts
            results[valid[0]] = _interpret(outs, threshold, id2label=id2label)
        else:
            # Fall back to per-pair if mock or custom pipe returned unexpected shape
            for i in valid:
                results[i] = _verify_one(pairs[i][0], pairs[i][1], pipe, threshold)
    except Exception as e:
        logger.warning("[Verification] Batch NLI failed (%s); falling back to per-pair", e)
        for i in valid:
            results[i] = _verify_one(pairs[i][0], pairs[i][1], pipe, threshold)

    return results


def _cached_batch(
    pairs: list[tuple[str, str]],
    pipe,
    threshold: float,
    cache_path: Optional[str] = None,
    batch_size: int = 16,
) -> list[tuple[str, Optional[float]]]:
    """
    Batch NLI with a disk cache (via shelve) so re-runs only compute new pairs.
    """
    if not pairs:
        return []

    if cache_path is None:
        cache_dir = getattr(
            config,
            "NLI_CACHE_DIR",
            os.path.join(Path(__file__).resolve().parent.parent, "data", "nli_cache"),
        )
        try:
            os.makedirs(cache_dir, exist_ok=True)
        except Exception:
            pass
        cache_path = os.path.join(cache_dir, "nli_cache")

    model_name = getattr(config, "NLI_MODEL_NAME", "cross-encoder/nli-deberta-v3-small")
    pipe_id = getattr(getattr(pipe, "model", None), "name_or_path", getattr(pipe, "__name__", "model"))

    def _pair_key(c: str, e: str) -> str:
        raw_key = f"{model_name}|{pipe_id}|{threshold:.2f}|{c.strip()}|{(e or '').strip()[:1024]}"
        return hashlib.sha1(raw_key.encode("utf-8")).hexdigest()

    results: list[Optional[tuple[str, Optional[float]]]] = [None] * len(pairs)
    todo_indices = []

    try:
        with shelve.open(cache_path) as db:
            for i, (claim, evidence) in enumerate(pairs):
                if not (claim or "").strip() or not (evidence or "").strip():
                    results[i] = (UNVERIFIED, None)
                    continue
                k = _pair_key(claim, evidence)
                cached_val = db.get(k)
                if cached_val is not None and isinstance(cached_val, (tuple, list)) and len(cached_val) == 2:
                    results[i] = (cached_val[0], cached_val[1])
                else:
                    todo_indices.append(i)

            if todo_indices:
                todo_pairs = [pairs[i] for i in todo_indices]
                computed = _verify_batch(todo_pairs, pipe, threshold, batch_size=batch_size)
                for i, verdict in zip(todo_indices, computed):
                    results[i] = verdict
                    try:
                        db[_pair_key(pairs[i][0], pairs[i][1])] = verdict
                    except Exception:
                        pass
    except Exception as err:
        logger.debug("[Verification] Disk cache read/write error (%s); running batch directly", err)
        return _verify_batch(pairs, pipe, threshold, batch_size=batch_size)

    return [(v if v is not None else (UNVERIFIED, None)) for v in results]


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def verification_agent(
    evidence_items: list,
) -> dict:
    """
    Verify all extracted evidence items against their source text.

    Parameters
    ----------
    evidence_items : list[dict]
        The list stored in ResearchState["evidence"].
        Each item must have "claim", "evidence", "source_url", "source_title".

    Returns
    -------
    dict  (VerificationSummary)
        results, counts, and the hallucination_rate headline metric.
    """
    results: list = []

    if not evidence_items:
        return _empty_summary()

    if not config.ENABLE_VERIFICATION:
        logger.info(
            "[Verification] Disabled by config (ENABLE_VERIFICATION=false). "
            "Marking all claims as UNVERIFIED."
        )
        for item in evidence_items:
            results.append(
                _make_verification_result(
                    item=item,
                    label=UNVERIFIED,
                    confidence=None,
                )
            )
        return _build_summary(results)

    pipe = _load_nli_pipeline()

    if pipe is None:
        # Graceful fallback — transformers not installed
        for item in evidence_items:
            results.append(
                _make_verification_result(
                    item=item,
                    label=UNVERIFIED,
                    confidence=None,
                )
            )
        return _build_summary(results)

    logger.info(
        "[Verification] Verifying %d claims with NLI (batch mode)...",
        len(evidence_items)
    )

    pairs = [
        (item.get("claim", ""), item.get("evidence", ""))
        for item in evidence_items
    ]
    verdicts = _cached_batch(
        pairs,
        pipe,
        config.NLI_CONFIDENCE_THRESHOLD,
    )

    for i, (item, (label, conf)) in enumerate(zip(evidence_items, verdicts)):
        results.append(
            _make_verification_result(
                item=item,
                label=label,
                confidence=conf,
            )
        )

        logger.debug(
            "[Verification] [%d/%d] %s (conf=%.2f): %s",
            i + 1, len(evidence_items),
            label, conf or 0.0, (item.get("claim") or "")[:80]
        )

    summary = _build_summary(results)

    logger.info(
        "[Verification] Grounding Audit Complete. "
        "Grounded (Entailed): %d (%.1f%%) | Unsupported (Neutral): %d (%.1f%%) | Contradicted: %d (%.1f%%)",
        summary["entailment_count"],
        (summary["grounded_rate"] or 0.0) * 100,
        summary["unsupported_count"],
        (summary["unsupported_rate"] or 0.0) * 100,
        summary["contradiction_count"],
        (summary["contradiction_rate"] or 0.0) * 100,
    )

    return summary


# ============================================================
# HELPERS
# ============================================================

def _build_summary(results: list) -> dict:
    total = len(results)

    if total == 0:
        return _empty_summary()

    entailment_count = sum(
        1 for r in results if r.get("label") == ENTAILMENT
    )
    neutral_count = sum(
        1 for r in results if r.get("label") == NEUTRAL
    )
    contradiction_count = sum(
        1 for r in results if r.get("label") == CONTRADICTION
    )
    unverified_count = sum(
        1 for r in results if r.get("label") == UNVERIFIED
    )

    # Only claims with an actual NLI classification are in this denominator.
    checked_count = entailment_count + neutral_count + contradiction_count

    grounded_rate = (
        entailment_count / checked_count if checked_count else None
    )
    unsupported_rate = (
        neutral_count / checked_count if checked_count else None
    )
    contradiction_rate = (
        contradiction_count / checked_count if checked_count else None
    )
    unsupported_or_contradicted_rate = (
        (neutral_count + contradiction_count) / checked_count
        if checked_count else None
    )

    return {
        "results": results,
        "total": total,
        "checked_count": checked_count,
        "unverified_count": unverified_count,

        "entailment_count": entailment_count,
        "grounded_count": entailment_count,
        "neutral_count": neutral_count,
        "unsupported_count": neutral_count,
        "contradiction_count": contradiction_count,

        "entailment_rate": (
            round(grounded_rate, 3) if grounded_rate is not None else None
        ),
        "grounded_rate": (
            round(grounded_rate, 3) if grounded_rate is not None else None
        ),
        "neutral_rate": (
            round(unsupported_rate, 3) if unsupported_rate is not None else None
        ),
        "unsupported_rate": (
            round(unsupported_rate, 3) if unsupported_rate is not None else None
        ),
        "contradiction_rate": (
            round(contradiction_rate, 3)
            if contradiction_rate is not None else None
        ),

        # Legacy metric retained for compatibility. It now counts both
        # neutral and contradicted claims, so it is not a confirmed
        # hallucination rate.
        "hallucination_rate": (
            round(unsupported_or_contradicted_rate, 3)
            if unsupported_or_contradicted_rate is not None else None
        ),
        "unsupported_or_contradicted_rate": (
            round(unsupported_or_contradicted_rate, 3)
            if unsupported_or_contradicted_rate is not None else None
        ),
    }


def _empty_summary() -> dict:
    return {
        "results": [],
        "checked_count": 0,
        "entailment_count": 0, "grounded_count": 0,
        "neutral_count": 0, "unsupported_count": 0,
        "contradiction_count": 0, "unverified_count": 0,
        "total": 0,
        "entailment_rate": None, "grounded_rate": None,
        "neutral_rate": None, "unsupported_rate": None,
        "contradiction_rate": None,
        "hallucination_rate": None,
        "unsupported_or_contradicted_rate": None,
    }


def get_unverified_claims(summary: dict) -> list:
    """
    Return items labelled NEUTRAL or CONTRADICTION for the report section.
    """
    return [
        r for r in summary.get("results", [])
        if r["label"] in (NEUTRAL, CONTRADICTION)
    ]


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    test_items = [
        {
            "claim":        "AI can diagnose cancer with 94% accuracy.",
            "evidence":     "Deep learning models achieved 94.5% accuracy on the "
                            "skin cancer detection benchmark, outperforming dermatologists.",
            "source_url":   "https://nature.com/example",
            "source_title": "Nature - AI Dermatology Study",
        },
        {
            "claim":        "No deaths have been attributed to AI diagnostic errors.",
            "evidence":     "Several cases of misdiagnosis by AI systems have been "
                            "documented in clinical settings, leading to patient harm.",
            "source_url":   "https://bmj.com/example",
            "source_title": "BMJ - AI Safety Report",
        },
        {
            "claim":        "Quantum computers will replace classical computers by 2030.",
            "evidence":     "The history of the Roman Empire spans over a thousand years.",
            "source_url":   "https://example.com/unrelated",
            "source_title": "Unrelated Source",
        },
    ]

    print("Running NLI verification on 3 test claims...")
    summary = verification_agent(test_items)
    print()
    for r in summary["results"]:
        conf_str = f"{r['confidence']:.2f}" if r["confidence"] else "N/A"
        print(f"[{r['label']}] (conf={conf_str}): {r['claim']}")
    print()
    print(f"Hallucination rate: {(summary['hallucination_rate'] or 0)*100:.1f}%")
