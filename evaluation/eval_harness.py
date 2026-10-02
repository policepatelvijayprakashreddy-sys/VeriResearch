"""
Evaluation Harness
==================

Scores a completed research report on multiple quality dimensions:

Rule-based (no LLM):
  - section_completeness   : all 7 required sections present
  - citation_coverage      : inline [N] markers / evidence items
  - source_diversity       : web sources vs academic sources
  - evidence_density       : evidence items per 1 000 report words
  - gap_acknowledgement    : knowledge-gaps section is non-trivial

LLM-as-judge (one Ollama call, optional):
  - factual_grounding      : claims supported by evidence  (1–10)
  - clarity                : readable, well-structured     (1–10)
  - depth                  : topic coverage thoroughness   (1–10)
  - critical_analysis      : quality of critique section   (1–10)
  - recommendation_quality : how actionable §6 is          (1–10)

Usage
-----
    from evaluation.eval_harness import run_eval, save_eval_report

    eval_report = run_eval(final_state, use_llm_judge=True)
    path = save_eval_report(eval_report, topic)
    print(format_eval_summary(eval_report))
"""

import logging
import os
import re
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ============================================================
# REQUIRED REPORT SECTIONS
# ============================================================

REQUIRED_SECTIONS = [
    "executive summary",
    "general research findings",
    "literature review",
    "critical analysis",
    "knowledge gaps",
    "recommendations",
    "conclusion",
]


# ============================================================
# PYDANTIC MODELS
# ============================================================

class LLMJudgeScore(BaseModel):
    """Structured output from the LLM-as-judge evaluation."""

    factual_grounding: int = Field(
        ge=1, le=10,
        description=(
            "Score 1-10: how well claims are supported by evidence. "
            "10 = every claim has a citation; 1 = mostly unsupported."
        )
    )

    clarity: int = Field(
        ge=1, le=10,
        description=(
            "Score 1-10: how readable and logically structured the report is. "
            "10 = excellent flow; 1 = confusing or disorganised."
        )
    )

    depth: int = Field(
        ge=1, le=10,
        description=(
            "Score 1-10: how thoroughly the topic is covered. "
            "10 = comprehensive; 1 = superficial."
        )
    )

    critical_analysis: int = Field(
        ge=1, le=10,
        description=(
            "Score 1-10: quality of the critical analysis section. "
            "10 = insightful critique; 1 = repeats literature review."
        )
    )

    recommendation_quality: int = Field(
        ge=1, le=10,
        description=(
            "Score 1-10: how specific and actionable the recommendations are. "
            "10 = concrete, evidence-based; 1 = vague."
        )
    )

    judge_reasoning: str = Field(
        description="Brief explanation of the scores given."
    )


class EvalReport(BaseModel):
    """Complete evaluation result for one research run."""

    topic: str
    timestamp: str

    # ── Rule-based metrics ────────────────────────────────────
    sections_found: list[str]
    sections_missing: list[str]
    section_completeness: float          # 0.0 – 1.0

    citation_count: int                  # inline [N] occurrences
    evidence_item_count: int
    citation_coverage: float             # 0.0 – 1.0

    web_source_count: int
    academic_source_count: int
    total_source_count: int
    source_diversity_ratio: float        # academic / total  (0.0–1.0)

    report_word_count: int
    evidence_density: float              # evidence items per 1 000 words

    gap_acknowledged: bool               # knowledge-gaps section non-trivial

    # ── LLM judge ────────────────────────────────────────────
    llm_judge: Optional[LLMJudgeScore] = None
    llm_judge_average: Optional[float] = None

    # ── Verification Node ────────────────────────────────────
    hallucination_rate: Optional[float] = None

    # ── Composite ────────────────────────────────────────────
    overall_score: float                 # 0.0 – 10.0


# ============================================================
# RULE-BASED METRICS
# ============================================================

def _check_sections(report_text: str) -> tuple[list[str], list[str]]:
    """Return (found, missing) section names."""
    text_lower = report_text.lower()
    found = [s for s in REQUIRED_SECTIONS if s in text_lower]
    missing = [s for s in REQUIRED_SECTIONS if s not in text_lower]
    return found, missing


def _get_report_body(report_text: str) -> str:
    """Extract report body text prior to consensus tables and sources lists."""
    for marker in [
        "## Cross-Paper Scientific Consensus",
        "# Cross-Paper Scientific Consensus",
        "## Source Quality & Verification Audit",
        "# Source Quality & Verification Audit",
        "### Sources",
        "## Sources",
        "# Sources",
        "## General Web Sources",
        "## Academic Literature Sources"
    ]:
        if marker in report_text:
            report_text = report_text.split(marker)[0]
    return report_text


def _count_citations(report_text: str) -> int:
    """Count inline citation markers like [1], [2], [12] in report body only."""
    body = _get_report_body(report_text)
    return len(re.findall(r'\[\d+\]', body))


def _citation_coverage(citation_count: int, evidence_count: int) -> float:
    if evidence_count == 0:
        return 0.0
    return min(citation_count / evidence_count, 1.0)


def _evidence_density(evidence_count: int, report_text: str) -> float:
    word_count = len(report_text.split())
    if word_count == 0:
        return 0.0
    return round((evidence_count / word_count) * 1000, 2)


def _gap_acknowledged(report_text: str) -> bool:
    """
    True if the knowledge-gaps section contains a meaningful paragraph
    (more than just a heading or a one-liner).
    """
    text_lower = report_text.lower()
    idx = text_lower.find("knowledge gap")
    if idx == -1:
        return False
    snippet = report_text[idx: idx + 600]
    return len(snippet.split()) > 40


def _source_diversity(web: int, academic: int) -> float:
    total = web + academic
    if total == 0:
        return 0.0
    return round(academic / total, 2)


# ============================================================
# COMPOSITE SCORE
# ============================================================

def _compute_overall(
    section_completeness: float,
    citation_coverage: float,
    source_diversity: float,
    evidence_density: float,
    gap_acknowledged: bool,
    llm_avg: Optional[float],
) -> float:
    """
    Weighted composite score 0–10.

    Rule-based  → 40 % weight
    LLM judge   → 60 % weight (if available), otherwise rule-based fills 100 %
    """

    # Normalise evidence density to 0–1 (cap at 10 items/1 000 words)
    density_score = min(evidence_density / 10.0, 1.0)

    rule_score = (
        section_completeness   * 0.35
        + citation_coverage    * 0.25
        + source_diversity     * 0.15
        + density_score        * 0.15
        + (0.10 if gap_acknowledged else 0.0)
    ) * 10.0  # scale to 0–10

    if llm_avg is not None:
        overall = 0.40 * rule_score + 0.60 * llm_avg
    else:
        overall = rule_score

    return round(overall, 2)


# ============================================================
# LLM JUDGE
# ============================================================

def _llm_judge(
    topic: str,
    report_text: str,
    evidence: list,
) -> Optional[LLMJudgeScore]:
    """
    Ask the local Ollama model to rate the report on 5 dimensions.
    Returns None on any failure so the rest of eval still works.
    """

    try:
        from llm_client import create_llm

        llm = create_llm()
        structured_llm = llm.with_structured_output(LLMJudgeScore)

        evidence_summary = "\n".join(
            f"- [{item.get('source_id', '')}] {item.get('claim', '')}"
            for item in (evidence or [])[:15]   # cap to keep prompt short
        ) or "No extracted evidence available."

        prompt = f"""You are an expert research-quality evaluator.

Research topic: {topic}

Evidence items used:
{evidence_summary}

Report to evaluate:
{report_text[:4000]}
{"... [truncated for evaluation]" if len(report_text) > 4000 else ""}

Score the report on exactly these 5 dimensions (integer 1–10 each):

1. factual_grounding  — claims supported by the provided evidence
2. clarity            — readable, well-structured prose
3. depth              — thoroughness of topic coverage
4. critical_analysis  — quality of the critique section
5. recommendation_quality — how specific and actionable the recommendations are

Also provide a brief judge_reasoning (2–4 sentences total).

Be honest and critical. Do not give all 10s.
Return all six fields."""

        result = structured_llm.invoke(prompt)
        return result

    except Exception as error:
        logger.warning("[Eval] LLM judge failed: %s", error)
        return None


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def run_eval(
    final_state: dict,
    use_llm_judge: bool = True,
) -> EvalReport:
    """
    Evaluate a completed research run.

    Parameters
    ----------
    final_state : dict
        The state dict returned by run_research().
    use_llm_judge : bool
        Whether to invoke the LLM-as-judge step (default True).

    Returns
    -------
    EvalReport
        Fully populated evaluation report.
    """

    topic          = final_state.get("topic", "Unknown")
    report_text    = final_state.get("final_report", "")
    web_sources    = final_state.get("web_sources", [])
    literature     = final_state.get("literature", {})
    academic_srcs  = literature.get("sources", [])
    evidence       = final_state.get("evidence", [])

    logger.info("[Eval] Running evaluation for topic: %s", topic)

    # ── Rule-based ────────────────────────────────────────────
    found, missing = _check_sections(report_text)
    section_completeness = len(found) / len(REQUIRED_SECTIONS)

    citation_count   = _count_citations(report_text)
    evidence_count   = len(evidence)
    cov              = _citation_coverage(citation_count, evidence_count)

    web_count        = len(web_sources)
    acad_count       = len(academic_srcs)
    diversity        = _source_diversity(web_count, acad_count)

    word_count       = len(report_text.split())
    density          = _evidence_density(evidence_count, report_text)

    gap_ok           = _gap_acknowledged(report_text)

    # ── LLM judge ────────────────────────────────────────────
    judge_result = None
    judge_avg    = None

    if use_llm_judge and report_text:
        logger.info("[Eval] Running LLM judge...")
        judge_result = _llm_judge(topic, report_text, evidence)

        if judge_result:
            scores = [
                judge_result.factual_grounding,
                judge_result.clarity,
                judge_result.depth,
                judge_result.critical_analysis,
                judge_result.recommendation_quality,
            ]
            judge_avg = round(sum(scores) / len(scores), 2)

    # ── Composite ────────────────────────────────────────────
    overall = _compute_overall(
        section_completeness,
        cov,
        diversity,
        density,
        gap_ok,
        judge_avg,
    )

    return EvalReport(
        topic=topic,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        sections_found=found,
        sections_missing=missing,
        section_completeness=round(section_completeness, 2),
        citation_count=citation_count,
        evidence_item_count=evidence_count,
        citation_coverage=round(cov, 2),
        web_source_count=web_count,
        academic_source_count=acad_count,
        total_source_count=web_count + acad_count,
        source_diversity_ratio=diversity,
        report_word_count=word_count,
        evidence_density=density,
        gap_acknowledged=gap_ok,
        llm_judge=judge_result,
        llm_judge_average=judge_avg,
        hallucination_rate=final_state.get("hallucination_rate"),
        overall_score=overall,
    )


# ============================================================
# FORMATTING
# ============================================================

def format_eval_summary(report: EvalReport) -> str:
    """One-line console summary of the eval result."""

    judge_str = (
        f"LLM judge: {report.llm_judge_average}/10 | "
        if report.llm_judge_average is not None
        else ""
    )

    section_status = (
        "OK"
        if not report.sections_missing
        else "MISSING: " + ", ".join(report.sections_missing)
    )

    gap_status = "YES" if report.gap_acknowledged else "NO"

    return (
        f"\n{'='*60}\n"
        f"  EVAL SCORE: {report.overall_score}/10\n"
        f"{'='*60}\n"
        f"  Sections   : {len(report.sections_found)}/{len(REQUIRED_SECTIONS)} "
        f"[{section_status}]\n"
        f"  Citations  : {report.citation_count} inline markers "
        f"({report.citation_coverage*100:.0f}% coverage)\n"
        f"  Sources    : {report.web_source_count} web + "
        f"{report.academic_source_count} academic\n"
        f"  Evidence   : {report.evidence_item_count} items "
        f"({report.evidence_density} per 1k words)\n"
        f"  Gap noted  : {gap_status}\n"
        f"  Contradiction Rate : {f'{report.hallucination_rate*100:.1f}%' if report.hallucination_rate is not None else 'N/A'}\n"
        f"  {judge_str}Overall: {report.overall_score}/10\n"
        f"{'='*60}"
    )


def save_eval_report(report: EvalReport, reports_dir: str) -> str:
    """
    Save the eval report as a markdown file in reports_dir.
    Returns the file path.
    """

    os.makedirs(reports_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = "".join(
        c if c.isalnum() or c in " _-" else "_"
        for c in report.topic
    ).strip().replace(" ", "_")[:60]

    filename  = f"{timestamp}_{safe_topic}_eval.md"
    filepath  = os.path.join(reports_dir, filename)

    # ── Build markdown ────────────────────────────────────────
    lines = [
        f"# Evaluation Report: {report.topic}",
        f"\n*Generated: {report.timestamp}*\n",
        "---\n",
        f"## Overall Score: {report.overall_score} / 10\n",

        "## Rule-Based Metrics\n",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Sections complete | {len(report.sections_found)}/{len(REQUIRED_SECTIONS)} |",
        f"| Missing sections | {', '.join(report.sections_missing) or 'None'} |",
        f"| Inline citations | {report.citation_count} |",
        f"| Citation coverage | {report.citation_coverage*100:.0f}% |",
        f"| Web sources | {report.web_source_count} |",
        f"| Academic sources | {report.academic_source_count} |",
        f"| Source diversity ratio | {report.source_diversity_ratio:.2f} |",
        f"| Report word count | {report.report_word_count:,} |",
        f"| Evidence items | {report.evidence_item_count} |",
        f"| Evidence density (per 1k words) | {report.evidence_density} |",
        f"| Knowledge gap acknowledged | {'Yes ✅' if report.gap_acknowledged else 'No ❌'} |",
        f"| NLI Hallucination Rate | {f'{report.hallucination_rate*100:.1f}%' if report.hallucination_rate is not None else 'Unverified'} |",
    ]

    if report.llm_judge:
        j = report.llm_judge
        lines += [
            "\n## LLM Judge Scores\n",
            "| Dimension | Score |",
            "|-----------|-------|",
            f"| Factual grounding | {j.factual_grounding}/10 |",
            f"| Clarity | {j.clarity}/10 |",
            f"| Depth | {j.depth}/10 |",
            f"| Critical analysis | {j.critical_analysis}/10 |",
            f"| Recommendation quality | {j.recommendation_quality}/10 |",
            f"| **Average** | **{report.llm_judge_average}/10** |",
            f"\n**Judge reasoning:** {j.judge_reasoning}",
        ]
    else:
        lines.append("\n*LLM judge was not run (use --no-eval to disable, or ensure Ollama is running).*")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("[Eval] Saved eval report to: %s", filepath)
    return filepath
