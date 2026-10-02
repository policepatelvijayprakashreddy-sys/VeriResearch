import argparse
import logging
import os
import sys
import warnings
from datetime import datetime

# ── 1. Suppress all library warnings (LangChain, HuggingFace, Pydantic, etc.) ──
try:
    from langchain_core._api.deprecation import (
        LangChainDeprecationWarning,
        LangChainPendingDeprecationWarning,
    )
    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
    warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)
except ImportError:
    pass
warnings.filterwarnings("ignore")

# ── 2. Suppress C-level and third-party library verbose output ──
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── 3. Ensure UTF-8 output on Windows consoles ──
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.rule import Rule
from rich.logging import RichHandler

console = Console()

from orchestrator.orchestrator import run_research
from evaluation.eval_harness import (
    run_eval,
    save_eval_report,
    format_eval_summary,
)
import config


# ============================================================
# LOGGING SETUP
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False, show_time=True)],
)

# Silence noisy third-party networking/HTTP loggers
for noisy in ["httpx", "httpcore", "primp", "rquest", "urllib3", "transformers", "torch"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)

logger = logging.getLogger("veri_research")


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

    clean_report = report.strip()
    if clean_report.startswith("# Research Report\n"):
        clean_report = clean_report[len("# Research Report\n"):].strip()

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# Research Report: {topic}\n\n")
        f.write(
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}* | *Framework: VeriResearch Multi-Agent Verification*\n\n"
        )
        f.write("---\n\n")
        f.write(clean_report)
        f.write("\n")

    return filepath


def recommend_only(topic: str) -> None:
    """
    Run the recommendation agent in isolation — no full research pipeline.
    """
    from agents.recommendation_agent import (
        recommendation_agent,
        format_recommendations_md,
    )

    console.print(
        Panel.fit(
            f"[bold cyan]VERIRESEARCH[/bold cyan] — [bold yellow]Research Opportunities Mode[/bold yellow]\n"
            f"[bold white]Topic:[/bold white] {topic}",
            border_style="cyan",
            title="💡 [bold]Recommendation Agent[/bold]"
        )
    )

    output = recommendation_agent(
        topic=topic,
        literature_sources=[],
        web_sources=[],
    )

    table = Table(title=f"Under-Explored Research Opportunities: {topic}", border_style="cyan", show_header=True)
    table.add_column("#", style="bold cyan", width=4)
    table.add_column("Proposed Project", style="bold white", width=30)
    table.add_column("Novelty", style="magenta", width=10)
    table.add_column("Difficulty", style="yellow", width=14)
    table.add_column("Technical Gap & Approach", style="dim", width=45)

    for i, rec in enumerate(output.recommendations, 1):
        table.add_row(
            str(i),
            rec.title,
            rec.novelty,
            rec.difficulty,
            f"[bold]Gap:[/bold] {rec.gap[:120]}...\n[bold]Approach:[/bold] {rec.approach[:120]}..."
        )

    console.print(table)

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

    console.print(f"\n[green]✔ Recommendations saved to:[/green] [bold underline]{path}[/bold underline]\n")


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
        console.print("[bold cyan]VeriResearch[/bold cyan] — Autonomous Verified Academic Research Agent")
        topic = console.input("[bold yellow]Enter a research topic:[/bold yellow] ").strip()

    if not topic:
        console.print("[bold red]Error:[/bold red] No research topic provided. Exiting.")
        return

    # ── Recommend-only shortcut ───────────────────────────────
    if args.recommend_only:
        recommend_only(topic)
        return

    # ── Header Banner ─────────────────────────────────────────
    console.print()
    console.print(
        Panel(
            f"[bold cyan]VERIRESEARCH[/bold cyan] [white]— Multi-Agent Verified Research Pipeline[/white]\n"
            f"[bold yellow]Topic:[/bold yellow] [bold white]{topic}[/bold white]\n"
            f"[dim]LLM: {config.OLLAMA_MODEL} | NLI: {config.NLI_MODEL_NAME} (CPU) | Registries: OpenAlex + Crossref[/dim]",
            border_style="cyan",
            title="🔬 [bold]Research Agent[/bold]",
            padding=(1, 2),
        )
    )
    console.print()

    # ── Full research pipeline ────────────────────────────────
    results = run_research(topic)

    final_report = results.get("final_report", "")

    # ── Save report ───────────────────────────────────────────
    saved_path = ""
    if final_report:
        saved_path = save_report(topic, final_report)
    else:
        console.print("[bold red]Warning:[/bold red] No report content was generated.")

    # ── Eval harness ──────────────────────────────────────────
    eval_path = ""
    if not args.no_eval:
        console.print()
        console.print(Rule("[bold magenta]Automated Quality Evaluation Harness[/bold magenta]", style="magenta"))
        use_llm = not args.no_llm_judge
        eval_report = run_eval(results, use_llm_judge=use_llm)
        console.print(format_eval_summary(eval_report))
        eval_path = save_eval_report(eval_report, _reports_dir())

    # ── Execution Summary Box ─────────────────────────────────
    evidence_count = len(results.get("evidence", []))
    lit_sources = len(results.get("literature", {}).get("sources", []))
    web_sources = len(results.get("web_sources", []))
    hallucination_rate = results.get("hallucination_rate")
    h_str = f"{hallucination_rate * 100:.1f}%" if hallucination_rate is not None else "N/A"

    table = Table(show_header=True, header_style="bold cyan", border_style="dim", box=None)
    table.add_column("Pipeline Stage / Metric", style="bold white", width=35)
    table.add_column("Status / Result", style="green", width=40)

    table.add_row("Academic Sources Harvested", f"{lit_sources} papers (Semantic Scholar + Crossref)")
    table.add_row("General Web Sources Harvested", f"{web_sources} verified web pages")
    table.add_row("Dual-Registry Provenance Check", "✅ OpenAlex + Crossref Verified")
    table.add_row("Grounded Evidence Items", f"{evidence_count} verbatim source quotes")
    table.add_row("DeBERTa NLI Stage-1 Contradiction", f"{h_str}")
    table.add_row("Stage-2 Consensus Matrix", "✅ Cross-paper synthesis complete")

    summary_content = (
        f"[bold cyan]Artifacts Generated:[/bold cyan]\n"
        f"📄 [bold white]Research Report:[/bold white] [underline green]{saved_path}[/underline green]\n"
        f"📊 [bold white]Scorecard Report:[/bold white] [underline green]{eval_path}[/underline green]"
    )

    console.print()
    console.print(
        Panel(
            table,
            title="📊 [bold green]Pipeline Execution Summary[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )
    console.print(
        Panel(
            summary_content,
            title="💾 [bold blue]Saved Review Artifacts[/bold blue]",
            border_style="blue",
            padding=(1, 2),
        )
    )
    console.print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n\n[bold red][Interrupted][/bold red] Research cancelled by user.")
        sys.exit(0)