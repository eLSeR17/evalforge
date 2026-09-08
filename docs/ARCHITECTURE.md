# Architecture

This document explains *why* evalforge is designed the way it is, how the
evaluation semantics generalize from the sibling portfolio projects, and the
exact contracts (quoting, source parsing, exit codes) the implementation
follows.

## 1. Problem and position in the portfolio

The user's portfolio story: *"I build agents (P1 `alpha-agent`), give them
verifiable memory (P2 `smart-contract-rag`), and guarantee their quality
(P3 `evalforge`)."*

P1 and P2 each ship their *own* eval harness embedded in their repo. evalforge
extracts the reusable essence of those harnesses into a **standalone,
subject-agnostic toolkit** — golden datasets + dual judge (deterministic
heuristic for CI + LLM-as-judge via Ollama) + a regression guard with semantic
exit codes — and closes the circle by evaluating the two siblings as subjects,
without copying or importing their code.

Hard constraints honoured by every component:

- **Separation**: evalforge never imports `alpha_agent` or `smart_contract_rag`;
  it invokes them from outside via `docker exec`. No confidential material is
  copied, referenced, or staged.
- **Hermetic tests**: the pytest suite never touches the network, Docker, or
  Ollama.
- **No threads**: all execution is sequential.

## 2. The golden dataset contract

A dataset is a JSON array of cases:

    id, topic, question, expected_keywords[], refuse(bool), doc_ids[]

Validation (fail-fast, with JSON paths to the offending field) enforces:

- unique, non-empty `id`; non-empty `question` and `topic`;
- `expected_keywords` **non-empty for answerable cases** and **empty for
  refusal cases** (traps have no expected content by construction);
- `doc_ids` entries are non-empty strings (optional provenance).

Generalisation note: P2 used `relevant_doc_id` (str|list) and `expect_answer`
(bool). evalforge uses `doc_ids` (always a list) and `refuse` (the inverse
framing: `True` = the subject must decline). Semantic content is identical;
naming is domain-free.

## 3. The eight metrics and how they generalize

Port of `smart_contract_rag/src/smart_contract_rag/evals/metrics.py` with
formulas preserved "where they apply" and the RAG domain stripped out:

| Metric | Reference formula (P2) | evalforge generalization |
|---|---|---|
| `faithfulness` | answer tokens in retrieved chunk text | answer tokens in the **golden reference context**: `expected_keywords` + `doc_ids` + exposed `sources` |
| `answer_relevance` | fraction of expected keywords in answer | identical formula |
| `citation_accuracy` | cited `doc_id` in expected `doc_id` | identical, `doc_id` = leading token of each source label |
| `context_precision` | relevant top-k / k | `retrieved_doc_ids` intersect `doc_ids` / `retrieved_doc_ids` |
| `context_recall` | retrieved relevant docs / relevant docs | identical |
| `answer_rate` | answered answerable / answerable | identical |
| `correct_refusal_rate` | correctness of the answer/refuse decision | identical |
| `hallucination_rate` | answered + ungrounded | answered where a refusal was expected, or answered with no exposed evidence |

**The `None` contract (documented departure from P2).** When a per-case metric
cannot be measured it is `None`, never a silent `0.0`. Aggregates average only
the measurable per-case values; a metric with no measurable case aggregates to
`None` and is **excluded from the verdict** (shown as `n/a` in the report). P2
instead treated unmeasurable metrics as a *vacuous pass* (`1.0`). evalforge
needs the stricter contract because subject capabilities vary: an agent that
exposes no retrieval must not be credited with perfect context recall.

**Heuristic faithfulness caveat.** With only CLI stdout visible, the heuristic
cannot see the sibling's retrieved chunk *text*; the reference-context
containment ratio is a conservative lexical proxy ("did the answer stay inside
the expected vocabulary + cited documents"). For semantic grounding evaluation
use `--judge ollama` (the LLM judge sees the full question/answer/keywords/doc
ids/sources). This is a documented trade-off of the standalone, read-only
design.

## 4. Subjects and the quoting contract

`Subject` is a one-method protocol (`ask(question) -> SubjectAnswer`).
`CliSubject` adapts a command line via `subprocess.run(capture_output=True,
text=True, timeout=...)`.

**Quoting robustness.** User questions can contain spaces, single/double
quotes, `$`, backticks, etc. Two placeholders are supported:

- `{question}` — **must be a whole argv element** (list-form subprocess, no
  shell involved). Embedding it inside a shell string raises `ValueError` at
  render time, because it would silently break quoting.
- `{question_b64}` — the question base64-encoded (UTF-8). The base64 alphabet
  `[A-Za-z0-9+/=]` cannot collide with shell syntax, so embedding it in a
  `sh -c` string is always safe. Preferred pattern:

      docker exec -e Q=<base64> <container> sh -c '... uses "$Q" ...'

Concrete registry patterns:

- **smart-contract-rag** (P2): the decoded question is passed as the CLI's
  positional argument:

      sh -c 'PYTHONPATH=/app/src python -m smart_contract_rag.cli "$(printf %s "$Q" | base64 -d)"'

  The inner `$(...)` is double-quoted command substitution, so the decoded text
  (spaces, newlines, quotes) becomes exactly one argv element.

- **alpha-agent** (P1): the question is piped on stdin into the wrapper from
  the subject's README, so there is **zero shell quoting** of the question
  itself (only the static Python wrapper is inside the `sh -c` string):

      printf %s "$Q" | base64 -d | PYTHONPATH=/repo/src python -c "<wrapper>"

  The wrapper imports `GuardedAlphaAgent`, `AlphaAgent`, `OllamaClient` and
  `TOOL_REGISTRY` from the mounted `/repo` and prints `final_answer`.

## 5. The dual judge

Both judges implement the `Judge` protocol (`score(case, answer) ->
PerCaseJudge`) and produce the same seven per-case signals.

- **HeuristicJudge** (default, `--judge heuristic`) composes the lexical
  metrics. Deterministic, dependency-free, CI-safe.
- **OllamaJudge** (`--judge ollama`) calls `{OLLAMA_URL}/api/chat` (default
  `http://ollama:11434`, the docker_default network — never `localhost`). The
  model scores faithfulness/relevance on a 0-5 scale and must reply with a
  single JSON object: `{"faithfulness": N, "relevance": N, "reasoning": "..."}`.
  The output is parsed with a defensive cascade (fenced block -> brace
  substring -> double-brace peel -> trailing-comma repair -> raw), so a
  slightly malformed response never crashes the eval. On any failure the
  deterministic scores are kept and `error` is populated. Each HTTP attempt
  builds a **fresh request** (never a reused one) — the retry lesson from the
  sibling evaluation pipelines.

## 6. Regression guard, verdict, and exit codes

`run_eval` computes the eight metrics and collapses them with
`RegressionThresholds` into a verdict using the same banding philosophy as P2's
`_verdict`: FAIL when a *measurable* metric breaches its floor/cap; WARN when a
metric sits within `warn_margin` of its guard; otherwise PASS.

Exit codes (semantic, for CI):

    PASS -> 0   (all measurable metrics above thresholds)
    FAIL -> 1   (a measurable metric breached a threshold, or usage error)
    WARN -> 0   (non-fatal in CI by design)

WARN is deliberately non-fatal. It means "a metric is approaching its guard —
review the report and decide." Making WARN fatal would let a single noisy metric
flip your whole pipeline red on an otherwise-healthy run and force unreviewed
threshold bumps. Use WARN for review, FAIL for hard gates.

Guards against misuse:

- an **empty dataset** raises `ValueError` (a guard with no cases is a
  configuration error, not a pass);
- a subject that **times out / crashes** marks that case as refused + errored
  (`error` populated) rather than aborting the run;
- an **empty or stale answer** is always mapped to a refusal by the runner,
  independently of the adapter's mark;
- metrics with no measurable case are excluded (see section 3).

## 7. Host-side E2E (`scripts/run_e2e.py`)

`run_e2e.py` is host-only and **never executed by pytest**. It:

1. Checks/creates the two subjects' containers:
   - `scr-rag-demo` must already exist (user-created per the sibling's
     `docs/LIVE_DEMO.md`); if missing it aborts with the exact creation
     command.
   - `alpha-agent-demo` is created on demand with the same pattern as
     `scr-rag-demo` / `python-lab` (`docker run -d --network docker_default
     -v <repo>:/repo python:3.12-slim sleep infinity`) and its dependencies
     installed the first time.
2. Loads the golden dataset, builds a `CliSubject` via the registry, runs the
   eval, and writes reports under `data/e2e/`.

## 8. Directory layout

    src/evalforge/    package (models, dataset, subjects, judges, metrics,
                      runner, report, cli)
    tests/            hermetic pytest suite
    scripts/          run_e2e.py (host-side, docker)
    data/golden/      alpha_agent.json, smart_contract_rag.json
    docs/             this document
