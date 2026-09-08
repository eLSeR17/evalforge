"""Report rendering: markdown for humans, JSON for CI consumers.

``render_markdown`` produces a standalone report with the header, the metrics
table, the per-case table, and the verdict section. ``write_report`` persists
both the markdown and the raw JSON artifact next to each other so CI can
consume either.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .models import EvalReport

#: Metrics displayed in the report table, in a stable order.
_METRIC_ORDER: tuple[str, ...] = (
    "faithfulness",
    "answer_relevance",
    "citation_accuracy",
    "context_precision",
    "context_recall",
    "answer_rate",
    "correct_refusal_rate",
    "hallucination_rate",
)

#: Metric name -> regression threshold key. ``correct_refusal_rate`` is
#: tracked and reported but intentionally not guarded in the report table
#: (its default agg is usually 1.0 and it is highly sensitive to trap count).
_THRESHOLD_KEY: dict[str, str] = {
    "faithfulness": "min_faithfulness",
    "answer_relevance": "min_answer_relevance",
    "citation_accuracy": "min_citation_accuracy",
    "context_precision": "min_context_precision",
    "context_recall": "min_context_recall",
    "answer_rate": "min_answer_rate",
    "hallucination_rate": "max_hallucination_rate",
}


def _fmt(value: float | None, fallback: str = "n/a") -> str:
    return f"{value:.4f}" if value is not None else fallback


def render_markdown(report: EvalReport) -> str:
    """Render *report* as a standalone markdown document."""
    lines = [
        "# Evalforge Evaluation Report",
        "",
        f"- **Subject**: `{report.subject}`",
        f"- **Judge**: `{report.judge}`",
        f"- **Date**: {report.created_at}",
        f"- **Cases**: {len(report.cases)}",
        f"- **Exit code**: {report.exit_code} (0 = pass, 1 = fail; WARN is non-fatal)",
        "",
        f"## Verdict: **{report.status}**",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]
    for name in _METRIC_ORDER:
        value = report.metrics.get(name)
        lines.append(f"| {name} | {_fmt(value)} |")

    lines += ["", "## Per-case", "", "| id | topic | answered | faith | rel | cit | p@k | rec | refusal | hallu | error |", "|----|-------|----------|-------|-----|-----|-----|-----|---------|-------|-------|"]
    for c in report.cases:
        lines.append(
            f"| {c.case_id} | {c.topic} | {c.answered} | "
            f"{_fmt(c.faith)} | {_fmt(c.rel)} | {_fmt(c.cit)} | {_fmt(c.p_at_k)} | "
            f"{_fmt(c.rec)} | {c.refusal_correct} | {_fmt(c.hallu, 'n/a')} | "
            f"{c.error.replace('|', '/')} |"
        )

    guarded = [
        name
        for name in _METRIC_ORDER
        if report.metrics.get(name) is not None
        and _THRESHOLD_KEY.get(name) is not None
        and _THRESHOLD_KEY[name] in report.thresholds
    ]
    if guarded:
        lines += ["", "## Regression guard", ""]
        lines.append("Thresholds applied to the measurable metrics:")
        lines.append("")
        lines.append("| Metric | Value | Threshold |")
        lines.append("|--------|-------|-----------|")
        for name in _METRIC_ORDER:
            key = _THRESHOLD_KEY.get(name)
            value = report.metrics.get(name)
            threshold = report.thresholds.get(key) if key else None
            if value is not None and threshold is not None:
                guard_is_max = key == "max_hallucination_rate"
                ok = value <= threshold if guard_is_max else value >= threshold
                marker = "OK" if ok else "BREACH"
                lines.append(f"| {name} | {_fmt(value)} | {threshold} | {marker} |")
    else:
        lines += [
            "",
            "## Regression guard",
            "",
            "No measurable metric was guarded in this run (all metrics n/a).",
        ]

    if report.metadata:
        meta = ", ".join(f"{k}={v}" for k, v in report.metadata.items() if not isinstance(v, dict))
        lines += ["", f"_Run metadata: {meta}_"]

    return "\n".join(lines)


def render_json(report: EvalReport) -> str:
    """Render *report* as a pretty-printed JSON artifact."""
    return json.dumps(report.to_dict(), indent=2, ensure_ascii=False)


def write_report(
    report: EvalReport,
    report_dir: str | Path,
    *,
    prefix: str = "eval_report",
    judge_label: str | None = None,
) -> tuple[Path, Path]:
    """Persist the markdown and JSON artifacts for *report*.

    Returns the ``(md_path, json_path)`` written. The filename embeds the
    subject, judge, and UTC timestamp so successive runs never collide.

    ``judge_label`` overrides the judge segment of the filename without
    changing ``report.judge`` — used by ``rejudge`` to write
    ``..._ollama-in-network_<ts>.{md,json}`` artifacts (the semantic run is
    Ollama, but only reachable inside the docker network). The value is
    sanitised for filesystem safety (no path separators).
    """
    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = report.created_at.replace(":", "").replace("-", "").replace("T", "T")
    judge_segment = report.judge if judge_label is None else judge_label
    judge_segment = _sanitize_filename_segment(judge_segment)
    base = out_dir / f"{prefix}_{report.subject}_{judge_segment}_{stamp}"
    md_path = base.with_suffix(".md")
    json_path = base.with_suffix(".json")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    json_path.write_text(render_json(report), encoding="utf-8")
    return md_path, json_path


def _sanitize_filename_segment(value: str) -> str:
    """Keep a filename segment filesystem-safe (``[A-Za-z0-9._-]``)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "judge"