import argparse
import logging
import os
from datetime import datetime

from orchestrator.orchestrator import run_research
from evaluation.eval_harness import (
    run_eval,
    save_eval_report,
    format_eval_summary,
)


# ============================================================
# LOGGING SETUP
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


# ============================================================
# HELPERS
# ============================================================

def _reports_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "reports")


def save_report(topic: str, report: str) -> str:
    """Save the final report to reports/ and return its path."""

    reports_dir = _reports_dir()
    os.makedirs(reports_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = "".join(
        c if c.isalnum() or c in " _-" else "_"
        for c in topic
    ).strip().replace(" ", "_")[:60]

    filepath = os.path.join(reports_dir, f"{timestamp}_{safe_topic}.md")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# Research Report: {topic}\n\n")
        f.write(
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
        )
        f.write("---\n\n")
        f.write(report)

    return filepath


def recommend_only(topic: str) -> None:
    """
    Run the recommendation agent in isolation — no full research pipeline.
    Useful for quickly exploring project ideas before committing to a full run.
    """

    from agents.recommendation_agent import (
        recommendation_agent,
        format_recommendations_md,
    )

    logger.info("Running recommendation-only mode for: %s", topic)

    output = recommendation_agent(
        topic=topic,
        literature_sources=[],
        web_sources=[],
    )

    print("\n\n")
    print("=" * 60)
    print("       RESEARCH PROJECT RECOMMENDATIONS")
    print("=" * 60)
    print(f"\nTopic: {topic}")
    print(f"\n{output.landscape_summary}\n")

    for i, rec in enumerate(output.recommendations, 1):
        print(f"\n{'─'*55}")
        print(f"{i}. {rec.title}")
        print(f"   Novelty: {rec.novelty}  |  Difficulty: {rec.difficulty}")
        print(f"   Gap: {rec.gap}")
        print(f"   Approach: {rec.approach}")

    # Save to reports/
    md = format_recommendations_md(output, topic)
    reports_dir = _reports_dir()
    os.makedirs(reports_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = "".join(
        c if c.isalnum() or c in " _-" else "_"
        for c in topic
    ).strip().replace(" ", "_")[:60]
    path = os.path.join(reports_dir, f"{timestamp}_{safe_topic}_recommendations.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Research Opportunities: {topic}\n\n")
        f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        f.write(md)

    print(f"\n[OK] Recommendations saved to: {path}\n")


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "VeriResearch — AI Research Agent\n"
            "Generates a fully cited research report and identifies "
            "under-explored project ideas."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--topic",
        type=str,
        default=None,
        help="Research topic (skips the interactive prompt).",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )

    parser.add_argument(
        "--no-eval",
        action="store_true",
        default=False,
        help="Skip the evaluation harness after the run.",
    )

    parser.add_argument(
        "--no-llm-judge",
        action="store_true",
        default=False,
        help=(
            "Run eval without the LLM-as-judge step "
            "(faster; only rule-based metrics)."
        ),
    )

    parser.add_argument(
        "--recommend-only",
        action="store_true",
        default=False,
        help=(
            "Skip the full research pipeline and just return "
            "project recommendations for the topic."
        ),
    )

    args = parser.parse_args()

    # Apply log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # ── Get topic ─────────────────────────────────────────────
    topic = args.topic
    if not topic:
        topic = input("Enter a research topic: ").strip()

    if not topic:
        logger.error("No research topic provided. Exiting.")
        return

    # ── Recommend-only shortcut ───────────────────────────────
    if args.recommend_only:
        recommend_only(topic)
        return

    # ── Full research pipeline ────────────────────────────────
    logger.info("Starting research on: %s", topic)

    results = run_research(topic)

    final_report = results.get("final_report", "")

    # ── Print report ──────────────────────────────────────────
    print("\n\n")
    print("=" * 60)
    print("                  FINAL RESEARCH REPORT")
    print("=" * 60)
    print(final_report)

    # ── Save report ───────────────────────────────────────────
    if final_report:
        saved_path = save_report(topic, final_report)
        print(f"\n[OK] Report saved to: {saved_path}")
        logger.info("Report saved to: %s", saved_path)
    else:
        logger.warning("No report content was generated.")

    # ── Eval harness ──────────────────────────────────────────
    if not args.no_eval:

        logger.info("Running evaluation harness...")

        use_llm = not args.no_llm_judge

        eval_report = run_eval(results, use_llm_judge=use_llm)

        # Console summary
        print(format_eval_summary(eval_report))

        # Save eval markdown
        eval_path = save_eval_report(eval_report, _reports_dir())
        print(f"[OK] Eval report saved to: {eval_path}\n")


if __name__ == "__main__":
    main()