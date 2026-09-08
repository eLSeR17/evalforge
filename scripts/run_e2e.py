#!/usr/bin/env python3
"""Host-side end-to-end orchestration: evaluates the portfolio sibling
projects (alpha-agent, smart-contract-rag) as evalforge subjects.

This script is **host-side only**: it talks to Docker on the host, uses the
real golden datasets, and writes the run artifacts under ``data/e2e/``. It is
NEVER executed by the unit test suite (which is fully hermetic).

How the subjects are wired
--------------------------
evalforge never imports the siblings' code. Each subject is a container that
is invoked from *outside* (``docker exec``) by the same
:class:`~evalforge.subjects.CliSubject` the CLI uses:

- ``smart-contract-rag``: uses the EXISTING ``scr-rag-demo`` container (repo
  mounted at ``/app``) — created previously by the user per the sibling's
  ``docs/LIVE_DEMO.md``.
- ``alpha-agent``: if the ``alpha-agent-demo`` container does not exist, it
  is created on demand with the same pattern as ``scr-rag-demo`` /
  ``python-lab``::

      docker run -d --name alpha-agent-demo --network docker_default \
        -v <path-to-alpha-agent>:/repo -w /repo python:3.12-slim sleep infinity

  and its Python dependencies are installed the first time
  (``pip install -r /repo/requirements.txt``).

Demo golden datasets:
    data/golden/alpha_agent.json        (10 cases)
    data/golden/smart_contract_rag.json (12 cases)

Usage::

    python3 scripts/run_e2e.py --subject all --judge heuristic
    python3 scripts/run_e2e.py --subject smart-contract-rag --judge ollama

Exit codes (same semantics as the CLI):
    0  PASS (WARN is non-fatal by design)
    1  FAIL or usage error
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SIBLINGS_ROOT = _REPO_ROOT.parent
_P1_REPO = _SIBLINGS_ROOT / "alpha-agent"

_ALPHA_CONTAINER = "alpha-agent-demo"
_SCR_CONTAINER = "scr-rag-demo"
_NETWORK = "docker_default"
_P1_MOUNT = "/repo"
_SCR_MOUNT = "/app"

_KNOWN_SUBJECTS = ("alpha-agent", "smart-contract-rag")


def _ensure_alpha_container(repo: Path) -> None:
    """Create + provision the alpha-agent-demo container when missing."""
    if _container_exists(_ALPHA_CONTAINER):
        return
    print(
        f"[e2e] creating container '{_ALPHA_CONTAINER}' "
        f"(network={_NETWORK}, repo mounted at {_P1_MOUNT})..."
    )
    subprocess.run(
        [
            "docker", "run", "-d", "--name", _ALPHA_CONTAINER,
            "--network", _NETWORK,
            "-v", f"{repo}:{_P1_MOUNT}",
            "-w", _P1_MOUNT,
            "python:3.12-slim", "sleep", "infinity",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    print("[e2e] installing alpha-agent dependencies (first run, this can take a while)...")
    try:
        subprocess.run(
            ["docker", "exec", _ALPHA_CONTAINER, "pip", "install", "--no-cache-dir", "-r", f"{_P1_MOUNT}/requirements.txt"],
            check=True,
            capture_output=True,
            text=True,
            timeout=900,
        )
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            "[e2e] ERROR: dependency install failed in the alpha-agent demo container.\n"
            f"  stderr: {exc.stderr[-800:]}\n"
            "  Fix the install, then re-run (the container already exists)."
        ) from exc
    print("[e2e] alpha-agent demo container ready.")


def _ensure_scr_container() -> None:
    """Verify the smart-contract-rag demo container exists (user-created)."""
    if not _container_exists(_SCR_CONTAINER):
        raise SystemExit(
            f"[e2e] ERROR: container '{_SCR_CONTAINER}' not found.\n"
            "  Create it once with (see the sibling's docs/LIVE_DEMO.md):\n\n"
            f"  cd <smart-contract-rag> && \\\n"
            "  docker run -d --name scr-rag-demo --network docker_default \\\n"
            f"    -v \"$PWD\":{_SCR_MOUNT} -w {_SCR_MOUNT} python:3.12-slim sleep infinity\n\n"
            "  then install deps and build the index (fetch_corpus + index_corpus)."
        )


def _container_exists(name: str) -> bool:
    proc = subprocess.run(
        ["docker", "ps", "-a", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return name in (proc.stdout or "").split()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subject",
        choices=(*_KNOWN_SUBJECTS, "all"),
        default="all",
        help="Subject(s) to evaluate (default: all).",
    )
    parser.add_argument(
        "--judge",
        choices=("heuristic", "ollama"),
        default="heuristic",
        help="Judge to use (default: heuristic — no Ollama needed).",
    )
    parser.add_argument(
        "--golden", default=None,
        help="Custom golden dataset path (default: data/golden/<snake_case subject>.json).",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OLLAMA_MODEL"),
        help="Ollama judge model (default: $OLLAMA_MODEL).",
    )
    parser.add_argument(
        "--report-dir",
        default=str(_REPO_ROOT / "data" / "e2e"),
        help="Output directory for report artifacts (default: data/e2e/).",
    )
    parser.add_argument(
        "--alpha-agent-repo",
        default=str(_P1_REPO),
        help="Path to the alpha-agent repo for the demo container mount.",
    )
    args = parser.parse_args(argv)

    sys.path.insert(0, str(_REPO_ROOT / "src"))

    from evalforge.dataset import EvalDataset, default_golden_filename
    from evalforge.judges import HeuristicJudge, OllamaJudge
    from evalforge.report import write_report
    from evalforge.runner import RegressionThresholds, run_eval
    from evalforge.subjects import build_subject

    # Default golden: data/golden/<snake_case subject>.json via the shared
    # naming convention (alpha-agent -> alpha_agent.json); --golden overrides.
    default_golden = {
        name: str(_REPO_ROOT / "data" / "golden" / default_golden_filename(name))
        for name in _KNOWN_SUBJECTS
    }

    subjects = list(_KNOWN_SUBJECTS) if args.subject == "all" else [args.subject]

    # Pre-flight container checks (fail fast with actionable messages).
    for name in subjects:
        if name == "alpha-agent":
            _ensure_alpha_container(Path(args.alpha_agent_repo))
        else:
            _ensure_scr_container()

    judge = HeuristicJudge()
    if args.judge == "ollama":
        judge = OllamaJudge(model=args.model or os.environ.get("OLLAMA_MODEL"))

    worst_exit = 0
    for name in subjects:
        golden = args.golden or default_golden[name]
        print(f"\n[e2e] evaluating '{name}' against {golden}")
        dataset = EvalDataset.from_json(golden)
        subject = build_subject(name)
        report = run_eval(
            subject,
            list(dataset),
            judge=judge,
            thresholds=RegressionThresholds(),
            dataset_source=dataset.source,
            metadata={"mode": "e2e-docker", "judge": args.judge},
        )
        md_path, json_path = write_report(report, args.report_dir)
        print(f"[e2e] {name}: verdict={report.status} exit_code={report.exit_code}")
        print(f"[e2e] artifacts: {md_path} / {json_path}")
        worst_exit = max(worst_exit, report.exit_code)

    print(f"\n[e2e] done. aggregate exit code: {worst_exit}")
    return worst_exit


if __name__ == "__main__":
    raise SystemExit(main())