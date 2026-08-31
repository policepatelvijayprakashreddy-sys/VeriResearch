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

import logging
from typing import List, Optional

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

    try:
        from transformers import pipeline as hf_pipeline
        from config import NLI_MODEL_NAME

        logger.info(
            "[Verification] Loading NLI model: %s "
            "(first run downloads ~185 MB)...",
            NLI_MODEL_NAME
        )

        _nli_pipeline = hf_pipeline(
            "zero-shot-classification",
            model=NLI_MODEL_NAME,
            device=-1,           # CPU
            multi_label=False,
        )

        logger.info("[Verification] NLI model loaded.")
        return _nli_pipeline

    except ImportError:
        logger.warning(
            "[Verification] transformers / torch not installed. "
            "Run: pip install transformers torch --index-url "
            "https://download.pytorch.org/whl/cpu\n"
            "All claims will be labelled UNVERIFIED."
        )
        return None

    except Exception as error:
        logger.error(
            "[Verification] Failed to load NLI model: %s. "
            "All claims will be labelled UNVERIFIED.", error
        )
        return None


# ============================================================
# VERIFY A SINGLE CLAIM
# ============================================================

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
        label      : ENTAILMENT | NEUTRAL | CONTRADICTION
        confidence : float 0.0-1.0 for the winning label
    """
    if not evidence.strip() or not claim.strip():
        return NEUTRAL, None

    try:
        # zero-shot-classification treats `sequence` as the premise
        # and `candidate_labels` as hypotheses to test
        candidate_labels = [ENTAILMENT, NEUTRAL, CONTRADICTION]

        # We frame it as: given this evidence snippet, does it
        # ENTAIL, stay NEUTRAL, or CONTRADICT the claim?
        result = pipe(
            sequences=evidence[:1024],          # truncate long snippets
            candidate_labels=candidate_labels,
            hypothesis_template=f"This text {{}}s the following: {claim[:256]}",
        )

        top_label      = result["labels"][0]
        top_confidence = result["scores"][0]

        # If confidence is below threshold, default to NEUTRAL
        if top_confidence < threshold:
            return NEUTRAL, top_confidence

        return top_label, top_confidence

    except Exception as error:
        logger.debug("[Verification] NLI call failed for one item: %s", error)
        return NEUTRAL, None


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
    from config import NLI_CONFIDENCE_THRESHOLD, ENABLE_VERIFICATION

    results: list = []

    if not evidence_items:
        return _empty_summary()

    if not ENABLE_VERIFICATION:
        logger.info(
            "[Verification] Disabled by config (ENABLE_VERIFICATION=false). "
            "Marking all claims as UNVERIFIED."
        )
        for item in evidence_items:
            results.append({
                "claim":        item.get("claim", ""),
                "evidence":     item.get("evidence", ""),
                "source_url":   item.get("source_url", ""),
                "source_title": item.get("source_title", ""),
                "label":        UNVERIFIED,
                "confidence":   None,
            })
        return _build_summary(results)

    pipe = _load_nli_pipeline()

    if pipe is None:
        # Graceful fallback — transformers not installed
        for item in evidence_items:
            results.append({
                "claim":        item.get("claim", ""),
                "evidence":     item.get("evidence", ""),
                "source_url":   item.get("source_url", ""),
                "source_title": item.get("source_title", ""),
                "label":        UNVERIFIED,
                "confidence":   None,
            })
        return _build_summary(results)

    logger.info(
        "[Verification] Verifying %d claims with NLI...",
        len(evidence_items)
    )

    for i, item in enumerate(evidence_items):
        claim    = item.get("claim", "")
        evidence = item.get("evidence", "")

        label, conf = _verify_one(
            claim, evidence, pipe, NLI_CONFIDENCE_THRESHOLD
        )

        results.append({
            "claim":        claim,
            "evidence":     evidence,
            "source_url":   item.get("source_url", ""),
            "source_title": item.get("source_title", ""),
            "label":        label,
            "confidence":   round(conf, 3) if conf is not None else None,
        })

        logger.debug(
            "[Verification] [%d/%d] %s (conf=%.2f): %s",
            i + 1, len(evidence_items),
            label, conf or 0.0, claim[:80]
        )

    summary = _build_summary(results)

    logger.info(
        "[Verification] Done. "
        "ENTAILMENT: %d | NEUTRAL: %d | CONTRADICTION: %d | "
        "Hallucination rate: %.1f%%",
        summary["entailment_count"],
        summary["neutral_count"],
        summary["contradiction_count"],
        (summary["hallucination_rate"] or 0.0) * 100,
    )

    return summary


# ============================================================
# HELPERS
# ============================================================

def _build_summary(results: list) -> dict:
    total = len(results)
    if total == 0:
        return _empty_summary()

    entailment_count    = sum(1 for r in results if r["label"] == ENTAILMENT)
    neutral_count       = sum(1 for r in results if r["label"] == NEUTRAL)
    contradiction_count = sum(1 for r in results if r["label"] == CONTRADICTION)
    unverified_count    = sum(1 for r in results if r["label"] == UNVERIFIED)

    verified = total - unverified_count

    hallucination_rate = (
        (neutral_count + contradiction_count) / verified
        if verified > 0
        else None
    )

    entailment_rate    = entailment_count    / verified if verified > 0 else None
    neutral_rate       = neutral_count       / verified if verified > 0 else None
    contradiction_rate = contradiction_count / verified if verified > 0 else None

    return {
        "results":             results,
        "entailment_count":    entailment_count,
        "neutral_count":       neutral_count,
        "contradiction_count": contradiction_count,
        "unverified_count":    unverified_count,
        "total":               total,
        "entailment_rate":     round(entailment_rate,    3) if entailment_rate    is not None else None,
        "neutral_rate":        round(neutral_rate,       3) if neutral_rate       is not None else None,
        "contradiction_rate":  round(contradiction_rate, 3) if contradiction_rate is not None else None,
        "hallucination_rate":  round(hallucination_rate, 3) if hallucination_rate is not None else None,
    }


def _empty_summary() -> dict:
    return {
        "results": [],
        "entailment_count": 0, "neutral_count": 0,
        "contradiction_count": 0, "unverified_count": 0,
        "total": 0,
        "entailment_rate": None, "neutral_rate": None,
        "contradiction_rate": None, "hallucination_rate": None,
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
