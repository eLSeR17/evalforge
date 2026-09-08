"""Command-line entry point: ``python -m evalforge.cli``.

Runs a golden dataset against a registered subject with the chosen judge and
applies the regression guard. CI-safe by default: the deterministic
``heuristic`` judge needs no network, no Docker, no Ollama.

Usage::

    python -m evalforge.cli --subject smart-contract-rag
    python -m evalforge.cli --subject alpha-agent --judge ollama --model qwen2.5:7b
    python -m evalforge.cli --subject smart-contract-rag --golden data/golden/custom.json --json

    # Re-score a previous run's artifact with a *semantic* judge (must run
    # inside the docker_default network, where http://ollama:11434 resolves):
    python -m evalforge.cli rejudge --artifact data/e2e/run.json \
        --judge ollama --golden data/golden/alpha_agent.json --model qwen2.5-coder:7b

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
import json
import os
import sys
from pathlib import Path

from .dataset import EvalDataset, default_golden_filename
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
    """Default golden path for a subject: ``data/golden/<snake_case>.json``."""
    return str(Path("data") / "golden" / default_golden_filename(subject))


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
        help="Path to the golden dataset JSON (default: data/golden/<snake_case subject>.json).",
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


def build_rejudge_parser() -> argparse.ArgumentParser:
    """Parser for the ``rejudge`` subcommand.

    Re-scores the captured responses of an existing evalforge artifact with a
    (semantic) judge, writing a **new** artifact — the original JSON is never
    modified. Designed to run inside a container on the ``docker_default``
    network, where ``http://ollama:11434`` resolves (see
    ``docs/ARCHITECTURE.md`` § "In-network semantic judging").
    """
    parser = argparse.ArgumentParser(
        prog="evalforge rejudge",
        description=(
            "Re-score a previous run's artifact with a judge (semantic by "
            "default: Ollama) without re-running the subject. Run this inside "
            "the docker_default network so the judge can reach Ollama."
        ),
    )
    parser.add_argument("--artifact", required=True, help="Path to the evalforge report JSON to re-score.")
    parser.add_argument(
        "--judge",
        choices=("heuristic", "ollama"),
        default="ollama",
        help="Judge to use (default: ollama — the semantic path).",
    )
    parser.add_argument(
        "--golden",
        default=None,
        help="Golden dataset JSON (default: the contract persisted in the artifact).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Ollama judge model (default: $OLLAMA_MODEL or qwen2.5-coder:7b).",
    )
    parser.add_argument(
        "--report-dir",
        default=None,
        help="Directory for the new artifacts (default: the artifact's own directory).",
    )
    parser.add_argument(
        "--no-warm-up",
        action="store_true",
        help="Skip the model warm-up call before scoring (default: warm up once).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the new report as JSON on stdout instead of markdown.",
    )
    return parser


def _main_rejudge(argv: list[str]) -> int:
    """Implementation of ``evalforge rejudge`` (see build_rejudge_parser)."""
    from .rejudge import rejudge_artifact

    args = build_rejudge_parser().parse_args(argv)

    try:
        artifact_path = Path(args.artifact)
        if not artifact_path.exists():
            raise FileNotFoundError(f"artifact not found: {artifact_path}")
        with artifact_path.open("r", encoding="utf-8") as fh:
            artifact = json.load(fh)
        if not isinstance(artifact, dict) or not isinstance(artifact.get("cases"), list):
            raise ValueError(f"{artifact_path} is not an evalforge report JSON")

        golden_cases = None
        if args.golden:
            golden_cases = {
                case.id: case
                for case in EvalDataset.from_json(args.golden)
            }

        if args.judge == "ollama":
            # Semantic judge model: explicit flag > $OLLAMA_MODEL > the coder
            # model validated on the live stack (see INC-004).
            judge = OllamaJudge(
                model=args.model
                or os.environ.get("OLLAMA_MODEL")
                or "qwen2.5-coder:7b"
            )
            if not args.no_warm_up:
                warm_up = getattr(judge, "warm_up", None)
                if warm_up:
                    ok, error = warm_up()
                    if ok:
                        print(
                            f"[rejudge] model {judge.model} warm (keep_alive={judge.keep_alive})",
                            file=sys.stderr,
                        )
                    else:
                        print(
                            f"[rejudge] warm-up failed ({error}); per-case fallback "
                            "contract still applies",
                            file=sys.stderr,
                        )
        else:
            judge = HeuristicJudge()

        outcome = rejudge_artifact(
            artifact,
            judge=judge,
            golden_cases=golden_cases,
            artifact_path=str(artifact_path),
        )
        report = outcome.report

        report_dir = args.report_dir or str(artifact_path.parent)
        # Semantic runs earn the "in-network" filename label so the artifact
        # name itself cannot be confused with the (host-side fallback) runs.
        judge_label = "ollama-in-network" if args.judge == "ollama" else None
        _md_path, json_path = write_report(
            report, report_dir, judge_label=judge_label
        )
    except (ValueError, KeyError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_CODE_FAIL

    print(f"\n[rejudge] re-scored {outcome.n_rejudged} of {len(report.cases)} cases "
          f"({outcome.n_skipped} skipped: {'; '.join(outcome.skipped_reasons) or 'none'})",
          file=sys.stderr)
    print(f"[rejudge] original artifact: {artifact_path}", file=sys.stderr)
    print(f"[rejudge] new artifact: {json_path}", file=sys.stderr)

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
    return report.exit_code


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Subcommand dispatch: `evalforge rejudge ...` -> rejudge; anything else
    # keeps the historical flat "eval" interface (`--subject ...`).
    if argv and argv[0] == "rejudge":
        return _main_rejudge(argv[1:])

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