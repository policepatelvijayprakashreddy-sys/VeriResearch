"""
Test Suite: Dual OpenAlex + Crossref Verification (5 Key Scenarios)
"""

import json
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

from search.source_quality_manager import verify_single_paper, batch_verify_papers
from agents.source_quality_agent import source_quality_agent, format_quality_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def run_benchmark():
    test_cases = [
        # Case 1: Normal Journal Paper
        {
            "id": "CASE_1_NORMAL_JOURNAL",
            "name": "Normal Journal Paper (NumPy in Nature)",
            "paper": {
                "title": "Array programming with NumPy",
                "doi": "10.1038/s41586-020-2649-2",
                "venue": "Nature",
                "url": "https://doi.org/10.1038/s41586-020-2649-2",
                "authors": ["Charles R. Harris", "K. Jarrod Millman", "Stéfan J. van der Walt"],
            },
            "expected_conf": "HIGH",
            "expected_retracted": False,
        },
        # Case 2: arXiv Preprint
        {
            "id": "CASE_2_ARXIV_PREPRINT",
            "name": "arXiv Preprint (Attention Is All You Need)",
            "paper": {
                "title": "Attention Is All You Need",
                "doi": "arXiv:1706.03762",
                "venue": "arXiv",
                "url": "https://arxiv.org/abs/1706.03762",
                "authors": ["Ashish Vaswani", "Noam Shazeer"],
            },
            "expected_conf": "MEDIUM",
            "expected_retracted": False,
        },
        # Case 3: Retracted Paper
        {
            "id": "CASE_3_RETRACTED_PAPER",
            "name": "Retracted Paper (Wakefield et al. in Lancet)",
            "paper": {
                "title": "RETRACTED: Ileal-lymphoid-nodular hyperplasia, non-specific colitis, and pervasive developmental disorder in children",
                "doi": "10.1016/S0140-6736(97)11096-0",
                "venue": "The Lancet",
                "url": "https://doi.org/10.1016/S0140-6736(97)11096-0",
                "authors": ["AJ Wakefield"],
            },
            "expected_conf": "RETRACTED_WARNING",
            "expected_retracted": True,
        },
        # Case 4: Invalid DOI
        {
            "id": "CASE_4_INVALID_DOI",
            "name": "Invalid / Non-Existent DOI",
            "paper": {
                "title": "A Fabricated Study on Imaginary Phenomena",
                "doi": "10.1000/invalid.doi.999999.notreal",
                "venue": "Non-Existent Journal",
                "url": "https://doi.org/10.1000/invalid.doi.999999.notreal",
                "authors": ["Nobody"],
            },
            "expected_conf": "LOW",
            "expected_retracted": False,
        },
        # Case 5: Metadata Mismatch
        {
            "id": "CASE_5_METADATA_MISMATCH",
            "name": "Metadata Mismatch (Real DOI with completely wrong title)",
            "paper": {
                "title": "Quantum Gravity and Black Hole Entropy in Holographic Spacetime",
                "doi": "10.1038/s41586-020-2649-2",  # Valid DOI for NumPy in Nature
                "venue": "Physical Review D",
                "url": "https://doi.org/10.1038/s41586-020-2649-2",
                "authors": ["Stephen Hawking"],
            },
            "expected_conf": "LOW",
            "expected_retracted": False,
        },
    ]

    print("\n" + "=" * 70)
    print("RUNNING 5 CORE BENCHMARK TEST CASES FOR SOURCE QUALITY VERIFICATION")
    print("=" * 70 + "\n")

    papers_list = [c["paper"] for c in test_cases]
    audit_results = source_quality_agent(papers_list)
    profiles = audit_results["profiles"]

    passed_count = 0
    total_cases = len(test_cases)

    for case, profile in zip(test_cases, profiles):
        case_id = case["id"]
        case_name = case["name"]
        exp_conf = case["expected_conf"]
        exp_ret = case["expected_retracted"]

        actual_conf = profile.get("identity_confidence")
        actual_ret = profile.get("is_retracted")
        oa_status = profile.get("openalex_status")
        cr_status = profile.get("crossref_status")
        reason = profile.get("confidence_reason")

        conf_match = (actual_conf == exp_conf)
        ret_match = (actual_ret == exp_ret)
        test_passed = conf_match and ret_match

        if test_passed:
            passed_count += 1
            status_symbol = "✅ PASS"
        else:
            status_symbol = "❌ FAIL"

        print(f"[{status_symbol}] {case_name}")
        print(f"       Expected: Identity={exp_conf}, Retracted={exp_ret}")
        print(f"       Actual:   Identity={actual_conf}, Retracted={actual_ret}")
        print(f"       OpenAlex: {oa_status} | Crossref: {cr_status}")
        print(f"       Reason:   {reason}\n")

    print("=" * 70)
    print(f"BENCHMARK SUMMARY: {passed_count}/{total_cases} CASES PASSED")
    print("=" * 70)

    print("\n" + audit_results["markdown_table"])
    print("\nRetraction Warnings Generated:")
    for warn in audit_results["retraction_warnings"]:
        print(f" - {warn}")

    assert passed_count == total_cases, f"Only {passed_count}/{total_cases} passed!"
    print("\nAll 5 verification test cases passed perfectly!")


if __name__ == "__main__":
    run_benchmark()
