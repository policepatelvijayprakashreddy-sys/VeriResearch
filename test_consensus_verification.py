"""
Test Suite: Cross-Paper Scientific Consensus Verification
========================================================
Tests 4 key scientific consensus scenarios:
  1. Consistent Support (corroborated by independent literature)
  2. Mixed Evidence (nuanced disagreement / differing experimental scopes)
  3. Retracted Paper Isolation (retracted papers flagged & excluded from positive consensus)
  4. Single Source Claim (lexical pre-filter skips irrelevant papers)
"""

import logging
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents.consensus_agent import consensus_agent, format_consensus_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def run_consensus_benchmark():
    print("\n" + "=" * 75)
    print("RUNNING CROSS-PAPER CONSENSUS VERIFICATION BENCHMARK")
    print("=" * 75 + "\n")

    # Mock evidence items (Stage 1 grounded outputs)
    evidence_items = [
        # Claim 1: High consensus empirical claim
        {
            "claim": "Array programming with NumPy improves vectorized numerical computing performance.",
            "evidence": "NumPy provides vectorization and array broadcasting that dramatically boosts execution speed in scientific computing.",
            "source_title": "Array Programming with NumPy",
            "source_url": "https://doi.org/10.1038/s41586-020-2649-2",
            "venue": "Nature",
        },
        # Claim 2: Mixed empirical evidence claim
        {
            "claim": "Dropout regularization eliminates overfitting on small medical imaging datasets.",
            "evidence": "Applying dropout to our deep convolutional network prevented overfitting on small clinical scan sets.",
            "source_title": "Deep Learning in Radiology",
            "source_url": "https://nature.com/example/radiology",
            "venue": "Nature Medicine",
        },
        # Claim 3: Retracted claim
        {
            "claim": "The MMR vaccine is linked to pervasive developmental disorders in children.",
            "evidence": "We identified gastrointestinal disease and developmental regression in children receiving the vaccine.",
            "source_title": "RETRACTED: Ileal-lymphoid-nodular hyperplasia in children",
            "source_url": "https://doi.org/10.1016/S0140-6736(97)11096-0",
            "venue": "The Lancet",
        },
        # Claim 4: Single source novel claim
        {
            "claim": "Cryogenic quantum-dot laser diodes achieve 99.8% modulation efficiency at 4 Kelvin.",
            "evidence": "Our custom cryogenic quantum-dot laser diode demonstrated 99.8% modulation efficiency at 4 Kelvin.",
            "source_title": "Cryogenic Quantum Optoelectronics",
            "source_url": "https://opticsexpress.org/example/quantum",
            "venue": "Optics Express",
        },
    ]

    # Stage 1 verification results (all verified as grounded)
    verification_results = [
        {"claim": item["claim"], "label": "ENTAILMENT"} for item in evidence_items
    ]

    # Independent academic literature sources in the system
    literature_sources = [
        # Originating source for Claim 1
        {
            "title": "Array Programming with NumPy",
            "url": "https://doi.org/10.1038/s41586-020-2649-2",
            "venue": "Nature",
            "content": "NumPy provides vectorization and array broadcasting that dramatically boosts execution speed in scientific computing.",
            "is_retracted": False,
        },
        # Independent Paper supporting Claim 1
        {
            "title": "High-Performance Vectorization in Modern Python",
            "url": "https://ieee.org/example/vectorization",
            "venue": "IEEE Computer",
            "content": "Vectorized array calculations using NumPy libraries deliver orders-of-magnitude speedups over pure Python loops.",
            "is_retracted": False,
        },
        # Independent Paper supporting Claim 1
        {
            "title": "Benchmarking Array Processing Frameworks",
            "url": "https://acm.org/example/benchmarks",
            "venue": "ACM Computing Surveys",
            "content": "Comparative evaluations confirm that NumPy array programming reduces memory overhead and improves runtime performance.",
            "is_retracted": False,
        },
        # Originating source for Claim 2
        {
            "title": "Deep Learning in Radiology",
            "url": "https://nature.com/example/radiology",
            "venue": "Nature Medicine",
            "content": "Applying dropout to our deep convolutional network prevented overfitting on small clinical scan sets.",
            "is_retracted": False,
        },
        # Independent Paper contradicting Claim 2
        {
            "title": "Limitations of Regularization in Small Clinical Cohorts",
            "url": "https://bmj.com/example/regularization",
            "venue": "The BMJ",
            "content": "Our clinical trial observed no statistically significant reduction in overfitting from dropout regularization on small patient cohorts.",
            "is_retracted": False,
        },
        # Originating source for Claim 3 (Retracted)
        {
            "title": "RETRACTED: Ileal-lymphoid-nodular hyperplasia in children",
            "url": "https://doi.org/10.1016/S0140-6736(97)11096-0",
            "venue": "The Lancet",
            "content": "We identified gastrointestinal disease and developmental regression in children receiving the vaccine.",
            "is_retracted": True,
        },
        # Independent Paper refuting Claim 3
        {
            "title": "Vaccine Safety and Pediatric Neurodevelopment: A Nationwide Study",
            "url": "https://nejm.org/example/vaccine-safety",
            "venue": "New England Journal of Medicine",
            "content": "Comprehensive epidemiological analyses demonstrate no causal association between MMR vaccination and autism spectrum disorders.",
            "is_retracted": False,
        },
        # Originating source for Claim 4
        {
            "title": "Cryogenic Quantum Optoelectronics",
            "url": "https://opticsexpress.org/example/quantum",
            "venue": "Optics Express",
            "content": "Our custom cryogenic quantum-dot laser diode demonstrated 99.8% modulation efficiency at 4 Kelvin.",
            "is_retracted": False,
        },
        # Completely irrelevant paper (Testing lexical pre-filter)
        {
            "title": "Historical Architecture of Roman Aqueducts and Water Reservoirs",
            "url": "https://cambridge.org/example/archaeology",
            "venue": "Cambridge Archaeological Journal",
            "content": "Analysis of ancient hydraulic mortar reveals sophisticated masonry techniques utilized in the Mediterranean basin.",
            "is_retracted": False,
        },
    ]

    def mock_nli_pipeline(inputs, *args, **kwargs):
        if isinstance(inputs, dict):
            text = (inputs.get("text") or "").lower()
            pair = (inputs.get("text_pair") or "").lower()
            seq_lower = f"{text} {pair}"
        else:
            seq_lower = str(inputs).lower()

        if "speedups" in seq_lower or "performance" in seq_lower or "prevented overfitting" in seq_lower:
            return [{"label": "ENTAILMENT", "score": 0.93}, {"label": "NEUTRAL", "score": 0.05}, {"label": "CONTRADICTION", "score": 0.02}]
        elif "no statistically significant" in seq_lower or "no causal association" in seq_lower:
            return [{"label": "CONTRADICTION", "score": 0.91}, {"label": "NEUTRAL", "score": 0.06}, {"label": "ENTAILMENT", "score": 0.03}]
        else:
            return [{"label": "NEUTRAL", "score": 0.85}, {"label": "ENTAILMENT", "score": 0.10}, {"label": "CONTRADICTION", "score": 0.05}]

    res = consensus_agent(
        evidence_items=evidence_items,
        verification_results=verification_results,
        literature_sources=literature_sources,
        max_claims=4,
        pipe=mock_nli_pipeline,
    )

    profiles = res["profiles"]
    expected_verdicts = [
        "CONSISTENT_SUPPORT",
        "MIXED_EVIDENCE",
        "PREDOMINANTLY_CONTRADICTED",
        "SINGLE_SOURCE_CLAIM",
    ]

    passed = 0
    for i, (p, exp_v) in enumerate(zip(profiles, expected_verdicts), start=1):
        actual_v = p.get("verdict")
        sup_count = len(p.get("supporting", []))
        con_count = len(p.get("contradicting", []))
        ret_count = len(p.get("retracted_alerts", []))
        reason = p.get("verdict_reason")

        is_match = (actual_v == exp_v)
        status = "✅ PASS" if is_match else "❌ FAIL"
        if is_match:
            passed += 1

        print(f"[{status}] Test Claim {i}: {p['claim'][:50]}...")
        print(f"       Expected: {exp_v}")
        print(f"       Actual:   {actual_v}")
        print(f"       Counts:   Supporting={sup_count} | Contradicting={con_count} | Retracted Alerts={ret_count}")
        print(f"       Reason:   {reason}\n")

    print("=" * 75)
    print(f"BENCHMARK RESULT: {passed}/{len(expected_verdicts)} CASES PASSED")
    print("=" * 75)

    print("\n" + res["markdown_table"])

    assert passed == len(expected_verdicts), f"Only {passed}/{len(expected_verdicts)} passed!"
    print("\nAll Cross-Paper Consensus benchmark test cases passed successfully!")


if __name__ == "__main__":
    run_consensus_benchmark()
