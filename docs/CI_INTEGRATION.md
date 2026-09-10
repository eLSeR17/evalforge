# EvalForge — CI integration

EvalForge is built around a regression guard with **semantic exit codes**, so
it drops into any pipeline as a plain command — no wrapper daemon, no API.

## The exit-code contract

| Exit | Meaning                                                                 |
|------|-------------------------------------------------------------------------|
| `0`  | **PASS** — every configured metric is at/above its threshold.           |
| `0`  | **WARN** — at least one metric is borderline. WARN is **non-fatal by
        design**: it must be seen by a human but must not brick the build. |
| `1`  | **FAIL** — a measurable metric breached its regression threshold (or a
        usage error — missing subject, bad golden, bad override).          |

That separation is the core design choice: **FAIL stops the pipeline,
WARN smells** (the report is written either way and the human reviews it).

## Wiring (native, no Action needed)

The simplest integration is a plain workflow step; the exit code does the
gate-keeping:

```yaml
- name: Regression guard (heuristic judge, hermetic)
  run: |
    pip install -e '.[dev]'
    evalforge --subject smart-contract-rag --judge heuristic
  # exit 1 (FAIL) fails the job; exit 0 (PASS/WARN) passes it.
```

Threshold overrides, either via the environment (shared by every invocation)
or repeatable CLI flags (per invocation, highest precedence):

```bash
export EVALFORGE_THRESHOLD=hallucination_rate=0.1,min_answer_rate=0.75
evalforge --subject alpha-agent --threshold citation_accuracy=0.8
```

Available overridable keys: the metrics produced by the dual judge
(faithfulness, relevance, citation accuracy, answer rate, hallucination
rate, refusal rate — see `src/evalforge/runner.py::RegressionThresholds`).
Pass a `--golden <path>` to point at a golden dataset that lives anywhere the
checkout can reach.

## Reusable Action (`action.yml`)

For consumers that want a one-step gate with artifact upload, EvalForge
ships a composite Action. Use it from **inside this repository** (or any
repo that checks it out):

```yaml
- name: EvalForge regression gate
  uses: ./
  with:
    subject: smart-contract-rag
    judge: heuristic          # default; ollama needs a reachable Ollama
    thresholds: "hallucination_rate=0.1"
```

The Action installs EvalForge, runs the guard, and uploads the report
directory as a workflow artifact (`data/reports`). It inherits the same
exit-code contract — a FAIL fails the calling job.

### Ollama judge in cloud CI — read this

`--judge ollama` only produces real semantic scores when `OLLAMA_URL` is
reachable **from where the job runs**. On GitHub-hosted runners there is no
Ollama, and by design EvalForge **falls back to the heuristic gate** rather
than erroring (see `docs/ARCHITECTURE.md`, judge fallback). Consequences:

- `judge: ollama` with no reachable Ollama runs the deterministic gate and
  exits per *its* thresholds — still a valid regression signal.
- Real LLM-as-judge runs belong on the docker network, as a manual/release
  step: `evalforge --subject <X> --judge ollama` inside `docker_default`
  (the pattern from `docs/LIVE_EVAL.md`).

## Guarding updates to the golden datasets

Golden datasets are the contract: a PR that changes
`data/golden/<subject>.json` changes the pass/fail bar for everyone. Treat
golden edits like test-fixture edits — review them with the same care as
code, and let the gate decide.

## Recommended CI layout

```yaml
jobs:
  test:      # hermetic unit suite (pytest) — false-positive canary
  lint:      # ruff — code hygiene
  security:  # pip-audit + gitleaks — dependency & secret scanning
  regression:# evalforge --judge heuristic — semantic regression gate (this repo)
```
