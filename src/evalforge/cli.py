"""Command-line entry point: ``python -m evalforge.cli``.

Runs a golden dataset against a registered subject with the chosen judge and
applies the regression guard. CI-safe by default: the deterministic
``heuristic`` judge needs no network, no Docker, no Ollama.

Usage::

    python -m evalforge.cli --subject smart-contract-rag
    python -m evalforge.cli --subject alpha-agent --judge ollama --model qwen2.5:7b
    python -m evalforge.cli --subject smart-contract-rag --golden data/golden/custom.json --json

Exit codes:
    0  PASS (and WARN — WARN is non-fatal in CI by design; review the report)
    1  FAIL (a measurable metric breached its regression threshold) or usage error

Environment:
    OLLAMA_URL           Ollama endpoint (default http://ollama:11434)
    OLLAMA_MODEL         Judge model (default qwen2.5:7b)
    EVALFORGE_THRESHOLD  Optional comma-separated ``key=value`` threshold
                         overrides (e.g. ``min_answer_rate=0.80``); the CLI
                         ``--threshold`` flag wins over the environment.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .dataset import EvalDataset
from .judges import HeuristicJudge, OllamaJudge
from .report import write_report
from .runner import EXIT_CODE_FAIL, EXIT_CODE_PASS, RegressionThresholds, run_eval
from .subjects import SUBJECTS, build_subject

_DEFAULT_REPORT_DIR = "data/reports"
_THRESHOLD_ENV = "EVALFORGE_THRESHOLD"


def _parse_threshold_overrides(raw: str | None) -> dict[str, float]:
    """Parse ``key=value[,key=value...]`` into a threshold override dict."""
    if not raw:
        return {}
    overrides: dict[str, float] = {}
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(
                f"invalid threshold override {token!r}: expected key=value"
            )
        key, value = token.split("=", 1)
        try:
            overrides[key.strip()] = float(value.strip())
        except ValueError:
            raise ValueError(
                f"invalid threshold value for {key.strip()!r}: {value.strip()!r}"
            ) from None
    return overrides


def _default_golden(subject: str) -> str:
    return str(Path("data") / "golden" / f"{subject}.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evalforge",
        description=(
            "Evaluate a subject against a golden dataset with a dual judge "
            "(deterministic heuristic for CI, or LLM-as-judge via Ollama) "
            "and a regression guard with semantic exit codes."
        ),
    )
    parser.add_argument("--subject", required=True, choices=sorted(SUBJECTS), help="Subject to evaluate.")
    parser.add_argument(
        "--golden",
        default=None,
        help="Path to the golden dataset JSON (default: data/golden/<subject>.json).",
    )
    parser.add_argument(
        "--judge",
        choices=("heuristic", "ollama"),
        default="heuristic",
        help="Judge to use (default: heuristic — deterministic, CI-safe).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Ollama judge model (default: $OLLAMA_MODEL or qwen2.5:7b).",
    )
    parser.add_argument(
        "--threshold",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a regression threshold (repeatable, e.g. --threshold min_answer_rate=0.8).",
    )
    parser.add_argument(
        "--report-dir",
        default=_DEFAULT_REPORT_DIR,
        help="Directory for the report artifacts (default: data/reports/).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the report as JSON on stdout instead of markdown.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        # Thresholds: defaults <- env overrides <- CLI overrides.
        env_overrides = _parse_threshold_overrides(os.environ.get(_THRESHOLD_ENV))
        cli_overrides = _parse_threshold_overrides(",".join(args.threshold))
        overrides = {**env_overrides, **cli_overrides}
        thresholds = RegressionThresholds().with_overrides(overrides)

        golden = args.golden or _default_golden(args.subject)
        dataset = EvalDataset.from_json(golden)
        subject = build_subject(args.subject)

        if args.judge == "ollama":
            judge = OllamaJudge(model=args.model or os.environ.get("OLLAMA_MODEL"))
        else:
            judge = HeuristicJudge()

        report = run_eval(
            subject,
            list(dataset),
            judge=judge,
            thresholds=thresholds,
            dataset_source=dataset.source,
            metadata={"golden": golden, "model": getattr(judge, "model", None)},
        )
    except (ValueError, KeyError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_CODE_FAIL

    _, _json_path = write_report(report, args.report_dir)

    if args.json:
        from .report import render_json

        print(render_json(report))
    else:
        from .report import render_markdown

        print(render_markdown(report))
        if report.status == "WARN":
            print(
                "\n[note] WARN is non-fatal in CI by design: "
                "review the report and calibrate thresholds if needed.",
                file=sys.stderr,
            )
    print(f"\n[artifacts] {_json_path}", file=sys.stderr)
    return report.exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())