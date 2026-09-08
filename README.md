# evalforge

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A standalone, reusable LLM evaluation toolkit: golden datasets + a dual judge
> (deterministic heuristic for CI + LLM-as-judge via local Ollama) + a
> regression guard with semantic exit codes. The demo evaluates the sibling
> portfolio projects (`alpha-agent`, `smart-contract-rag`) as black-box
> subjects — no code copied, no internals imported.

## Problem

Every serious LLM pipeline ships some kind of eval — but the eval harness is
usually *bolted into* the project: domain-specific schemas, hard-wired metrics,
a single judge, no exit-code contract for CI. When you build a second or third
agent, you discover the harness is not reusable. And when you want to compare
two systems side by side, there is no common measuring stick.

## Solution

evalforge is the measuring stick, extracted and generalized from the eval
pipelines of the two sibling portfolio projects. It gives you:

- **Golden datasets** — a versioned, validated JSON contract
  (`id`, `topic`, `question`, `expected_keywords`, `refuse`, `doc_ids`) with
  fail-fast schema validation reporting the exact JSON path of every issue.
- **A dual judge** —
  `HeuristicJudge` (deterministic token-overlap, zero network, CI-safe) and
  `OllamaJudge` (structured 0-5 LLM-as-judge via `http://ollama:11434`, with a
  defensive JSON-parse cascade and graceful fallback to the heuristic).
- **A regression guard** — eight generalized metrics, PASS/WARN/FAIL verdict
  with warn bands, and a semantic exit code (`0` pass, `1` fail; WARN is
  non-fatal by design) so CI can gate on quality regressions.
- **Black-box subjects** — anything exposing `ask(question) -> answer` can be
  evaluated. The registry shows how to adapt a CLI in a Docker container with a
  shell-safe base64 quoting pattern.

The demo that closes the circle evaluates the other two portfolio projects as
subjects: **`alpha-agent`** (an investment-research agent with tools) and
**`smart-contract-rag`** (a RAG over smart-contract audits).

## Demo: evaluating the sibling projects

```bash
# 1. Heuristic judge (no Ollama needed, CI-safe)
python3 scripts/run_e2e.py --subject all --judge heuristic

# 2. LLM-as-judge via local Ollama
python3 scripts/run_e2e.py --subject smart-contract-rag --judge ollama

# 3. Run the CLI directly against one subject
python3 -m evalforge.cli --subject alpha-agent
```

### What a run produces

`scripts/run_e2e.py` evaluates each subject against its golden dataset and
writes the artifacts to `data/e2e/` (markdown + JSON; the CLI writes the same
artifacts to `data/reports/`). The markdown report is a single self-contained
document: a header (subject, judge, date, case count, exit code), the verdict,
the eight-metric table, a per-case detail table, and the regression-guard
section showing which thresholds were applied and which metrics breached:

```text
# Evalforge Evaluation Report

- **Subject**: `smart-contract-rag`
- **Judge**: `heuristic`
- **Cases**: 12
- **Exit code**: 0 (0 = pass, 1 = fail; WARN is non-fatal)

## Verdict: **FAIL**

## Metrics

| Metric | Value |
|--------|-------|
| faithfulness         | <value> |
| answer_relevance     | <value> |
| citation_accuracy    | <value> |
| context_precision    | <value> |
| context_recall       | <value> |
| answer_rate          | <value> |
| correct_refusal_rate | <value> |
| hallucination_rate   | <value> |

## Regression guard

| Metric | Value | Threshold | Status |
```

*Example output — replace with real Fase 2 numbers.* The `<value>` placeholders
illustrate the report *format* only; they are not measured results.

**Status — Fase 2 pending.** The toolkit itself is implemented and verified by
a deterministic unit suite (197 tests, no network/docker/Ollama). The real
end-to-end results against the sibling subjects have **not** been executed yet;
they will be published here when the Fase 2 e2e run is done.

A FAIL verdict is not a bug — it is the regression guard doing its job: the
report flags when a subject does not meet the configured golden thresholds on
that run (e.g. a `context_precision` below its floor). You interpret the
breach, fix the subject or calibrate the thresholds
(`--threshold key=value`), and re-run.

> For reference, the sibling `smart-contract-rag` repo (P2) publishes results
> from **its own** eval harness (LLM-as-judge, 14 cases): `faithfulness
> 0.6667`, `answer_relevance 0.6444`, `citation_accuracy 0.4444`,
> `context_precision 0.1667`, `context_recall 0.5`, `answer_rate 0.75`,
> `correct_refusal_rate 0.7857`, `hallucination_rate 0.0`. Those are P2's own
> harness numbers — **not** evalforge output. Reproducing them from outside
> (black-box, without importing the siblings' code) is exactly what the
> evalforge e2e (Fase 2) will demonstrate.

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full design rationale:

- how the eight metrics generalize from the sibling `smart-contract-rag` evals,
- the `None` contract (unmeasurable metrics are excluded, never silently 0),
- the base64 quoting contract for subprocess subjects,
- the source/citation parsing contract, verified against the real CLI output,
- the WARN-is-non-fatal exit-code policy.

## Metrics

| Metric | What it measures |
|---|---|
| `faithfulness` | answer stays inside the golden reference context (keywords + doc ids + cited sources) |
| `answer_relevance` | expected keywords present in the answer |
| `citation_accuracy` | the answer cites an expected document |
| `context_precision` (p@k) | retrieved docs that are relevant |
| `context_recall` | relevant docs that were retrieved |
| `answer_rate` | answerable questions actually answered |
| `correct_refusal_rate` | refusal decisions that were right (traps refused, answerable answered) |
| `hallucination_rate` | fabricated responses (answered a trap, or answered with no evidence) |

## Stack

- **Python 3.11+** — type-hinted, dataclass models, no threads
- **pytest 8** — fully hermetic test suite (197 deterministic tests, no network/docker/Ollama)
- **requests** — the only runtime dependency (Ollama judge HTTP)
- **Ollama** (local) — optional LLM-as-judge at `http://ollama:11434`
- **Docker** — host-side demo (`docker exec` into the subjects' containers)

## Getting started

### Prerequisites

- Python 3.11+ (3.14 used for development)
- `pip install -e .[dev]` (the `dev` extra installs pytest, used by the test suite)
- For the E2E demo only: Docker + the `scr-rag-demo` container (and Ollama if
  you want `--judge ollama`). The unit tests need **none** of these.

### Run the tests

```bash
pytest
```

### Use it as a library

```python
from evalforge.dataset import EvalDataset
from evalforge.judges import HeuristicJudge
from evalforge.runner import run_eval
from evalforge.subjects import build_subject

dataset = EvalDataset.from_json("data/golden/alpha_agent.json")
report = run_eval(
    build_subject("alpha-agent"),
    list(dataset),
    judge=HeuristicJudge(),
)
print(report.status, report.exit_code)   # e.g. PASS 0
```

### CLI

```bash
python3 -m evalforge.cli --subject smart-contract-rag                    # heuristic, CI-safe
python3 -m evalforge.cli --subject alpha-agent --judge ollama --model qwen2.5:7b
python3 -m evalforge.cli --subject smart-contract-rag --json            # JSON artifact on stdout
python3 -m evalforge.cli --subject smart-contract-rag --threshold min_context_recall=0.4
```

Exit codes: `0` PASS (WARN is non-fatal), `1` FAIL or usage error. Reports are
written to `data/reports/` (markdown + JSON).

## Adding a new subject

1. Write a golden dataset under `data/golden/<name>.json`.
2. Register a `CliSubject` in `src/evalforge/subjects.py` with the
   `{question_b64}` pattern (or provide a `Subject` object and call
   `run_eval` directly).
3. Run `python3 -m evalforge.cli --subject <name>`.

## Golden datasets

- `data/golden/alpha_agent.json` — 10 finance-domain cases for `alpha-agent`
  (7 answerable with expected keywords, 1 grounded refusal, 2 traps).
- `data/golden/smart_contract_rag.json` — 12 smart-contract-audit cases for
  `smart-contract-rag` (9 answerable + canonical doc ids, 1 grounded refusal,
  2 traps).

## Limitations

- **Heuristic faithfulness is a lexical proxy** — it measures containment in
  the golden reference context, not semantic grounding (the LLM judge covers
  semantic scoring).
- **Read-only evaluation** — the CLI sees only what the subject prints
  (stdout): retrieval text is not visible, which is why
  context_precision/recall are `n/a` for subjects that expose no retrieval.
- **Live demo depends on the subjects' real containers and LLM** — results can
  vary with the model/tool state; deterministic behavior lives in the pytest
  suite.
- **Educational scope** — a portfolio demonstration of evaluation engineering,
  not a replacement for commercial eval platforms.

## License

MIT — see [LICENSE](LICENSE).