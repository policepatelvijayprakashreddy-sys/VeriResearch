"""
Benchmark Runner
================

Runs the pipeline across a curated dataset of topics, running once WITH
and once WITHOUT the Verification Agent to demonstrate its impact on
the `hallucination_rate` and `factual_grounding` score.

Usage
-----
    python -m evaluation.benchmark_runner          # run all topics
    python -m evaluation.benchmark_runner --quick  # run only 2 topics
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime

import config
from main import run_research
from evaluation.eval_harness import run_eval, save_eval_report

logger = logging.getLogger(__name__)

TOPICS_FILE = os.path.join(os.path.dirname(__file__), "benchmark_topics.json")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")


def load_topics(quick: bool = False) -> list:
    if not os.path.exists(TOPICS_FILE):
        logger.error(f"Missing topics file: {TOPICS_FILE}")
        return []
        
    with open(TOPICS_FILE, "r", encoding="utf-8") as f:
        topics = json.load(f)
        
    if quick:
        return topics[:2]
    return topics


def run_ablation(topic_data: dict) -> dict:
    """Run pipeline twice for the given topic: verification OFF vs ON."""
    
    topic = topic_data["topic"]
    topic_type = topic_data["type"]
    
    logger.info(f"\n{'='*60}\nBENCHMARK: {topic} [{topic_type}]\n{'='*60}")
    
    results = {"topic": topic, "type": topic_type, "runs": {}}
    
    for enable_verification in [False, True]:
        config.ENABLE_VERIFICATION = enable_verification
        mode = "VERIFICATION_ON" if enable_verification else "VERIFICATION_OFF"
        
        logger.info(f"\n--- Running: {mode} ---")
        start_time = time.time()
        
        try:
            # Force cache to True to speed up second run
            config.SEARCH_CACHE_ENABLED = True
            
            state = run_research(topic)
            eval_report = run_eval(state, use_llm_judge=True)
            
            # Save the individual run report
            save_eval_report(eval_report, RESULTS_DIR)
            
            duration = time.time() - start_time
            
            results["runs"][mode] = {
                "overall_score": eval_report.overall_score,
                "factual_grounding": eval_report.llm_judge.factual_grounding if eval_report.llm_judge else None,
                "hallucination_rate": eval_report.hallucination_rate,
                "duration_seconds": round(duration, 1)
            }
            
            logger.info(f"[{mode}] Score: {eval_report.overall_score}/10, "
                        f"Hallucination: {eval_report.hallucination_rate or 0:.1%} "
                        f"({duration:.1f}s)")
                        
        except Exception as error:
            logger.error(f"[{mode}] Run failed: {error}")
            results["runs"][mode] = None
            
    return results


def write_summary(all_results: list) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(RESULTS_DIR, f"{timestamp}_ablation_summary.md")
    
    lines = [
        "# Verification Agent Ablation Results",
        f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        "\nThis report compares the research pipeline's factual grounding score "
        "and hallucination rate with the Verification Agent turned OFF vs ON.\n",
        "| Topic (Type) | OFF Overall | ON Overall | OFF Factual Grounding | ON Factual Grounding | ON Hallucination Rate |",
        "|--------------|-------------|------------|-----------------------|----------------------|-----------------------|"
    ]
    
    for res in all_results:
        runs = res.get("runs", {})
        off = runs.get("VERIFICATION_OFF") or {}
        on = runs.get("VERIFICATION_ON") or {}
        
        topic_disp = res['topic']
        if len(topic_disp) > 40:
            topic_disp = topic_disp[:37] + "..."
            
        off_overall = off.get("overall_score", "FAIL")
        on_overall = on.get("overall_score", "FAIL")
        
        off_fg = off.get("factual_grounding", "N/A")
        on_fg = on.get("factual_grounding", "N/A")
        
        hr = on.get("hallucination_rate")
        hr_disp = f"{hr*100:.1f}%" if hr is not None else "N/A"
        
        lines.append(
            f"| {topic_disp} ({res['type']}) | "
            f"{off_overall} | {on_overall} | "
            f"{off_fg}/10 | {on_fg}/10 | "
            f"{hr_disp} |"
        )
        
    lines.append("\n**Conclusion:** Does the objective NLI hallucination rate correlate with the LLM's self-graded Factual Grounding score?")
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        
    return filepath


def main():
    parser = argparse.ArgumentParser(description="Run research ablation benchmarks")
    parser.add_argument("--quick", action="store_true", help="Run only the first 2 topics")
    parser.add_argument("--log-level", default="INFO", help="Set logging level")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    topics = load_topics(quick=args.quick)
    if not topics:
        return
        
    all_results = []
    
    for idx, topic_data in enumerate(topics, 1):
        logger.info(f"\nProcessing topic {idx}/{len(topics)}...")
        res = run_ablation(topic_data)
        all_results.append(res)
        
    summary_file = write_summary(all_results)
    print(f"\n\n{'='*40}\nABLATION COMPLETE\n{'='*40}")
    print(f"Summary written to: {summary_file}")


if __name__ == "__main__":
    main()
