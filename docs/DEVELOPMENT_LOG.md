# evalforge — Development Log

> A practical, honest account of how evalforge was built and validated: every
> incident below is real, reproduced, and verified. Dates refer to the build
> and e2e session of **2026-09-08**, with the ciclo A fixes and the ciclo B
> semantic closure landing on **2026-09-09**.

---

## 1. Overview

evalforge is the standalone measuring stick extracted from the eval pipelines
of the sibling portfolio projects: golden datasets + a dual judge
(deterministic heuristic for CI, LLM-as-judge for semantics) + a regression
guard with semantic exit codes. It closes the portfolio circle by evaluating
`alpha-agent` (P1) and `smart-contract-rag` (P2) as black-box subjects.

| Phase | Goal | Deliverables | Status |
|-------|------|--------------|--------|
| **Phase 1** | Reusable eval toolkit | `src/evalforge/` (models, dataset, subjects, judges, metrics, runner, report, cli) + golden datasets (10 + 12 cases) + `scripts/run_e2e.py` | ✅ |
| **Phase 2** | Golden-path hardening | Shared kebab→snake golden filename resolution (INC-001) — CLI + e2e resolve the same way; suite 197 → 204 tests | ✅ |
| **Phase 3** | Real e2e + CI | 4 live runs against the sibling containers (heuristic + ollama judges), `docs/LIVE_EVAL.md` canonical record, `.github/workflows/ci.yml` (3.11/3.12) | ✅ |
| **Phase 3, ciclo A** | Honest labeling + in-network semantic path | **INC-004**: fixes the misleading "ollama judge" labeling of the first e2e (the judge never connected from the host) — honest docs, `subject_answer` persisted in artifacts, `evalforge rejudge` CLI (in-network re-scoring), OllamaJudge hardening (timeout 180s, keep_alive 5m, warm-up); suite 204 → 223 tests | ✅ 2026-09-09 |
| **Phase 3, ciclo B** | Semantic LLM-as-judge executed in-network | **INC-004 RESOLVED**: the two `rejudge` runs ran for real inside `docker_default` (`qwen2.5-coder:7b`) on the 2026-09-08 captures — `eval_report_alpha-agent_ollama-in-network_20260908T232826Z` (10/10 re-judged, 0 skips) and `eval_report_smart-contract-rag_ollama-in-network_20260908T232921Z` (11/12, sr-001 skipped), **0 fallbacks**; README + `LIVE_EVAL.md` updated with the final semantic numbers and the subject×judge table | ✅ 2026-09-09 |

**Closing state**: 223 hermetic unit tests (no network/docker/Ollama) + 4 real
e2e runs on 2026-09-08, all with verdict FAIL (exit 1) — the regression guard
firing on real measured gaps, as designed. Two runs are the deterministic
heuristic gate; two are the **semantic LLM-as-judge executed in-network**
(`evalforge rejudge`, `qwen2.5-coder:7b`, 0 fallbacks; `sr-001` honestly
excluded because the subject timed out in the original run). INC-004 is
closed: every number in the README and `LIVE_EVAL.md` semantic sections comes
byte-for-byte from the `..._ollama-in-network_...` artifacts.

## 2. Development timeline

| Date | Phase | Milestone | Status |
|------|-------|-----------|--------|
| 2026-09-08 | 1 | evalforge core lands (initial commit `f78f590`): golden set, dual judge, regression guard, CLI, e2e script | ✅ |
| 2026-09-08 | 2 | **INC-001**: default golden path resolution broken for kebab-case subjects → fixed in `21f6ecb` (shared helper), 7 new tests (197 → 204) | ✅ fixed |
| 2026-09-08 | 3 | Real e2e #1 (alpha-agent heuristic) and #2 (smart-contract-rag heuristic): FAIL as designed, retrieval misses visible | ✅ |
| 2026-09-08 | 3 | Real e2e #3/#4 (ollama judge): **INC-002** (sr-001 subject timeout, operational) and dual-judge fallback observed live | ✅ diagnosed |
| 2026-09-08 | 3 | **INC-003** (design): heuristic faithfulness is brutally low on free text → dual-judge design documented, verdicts consistent across judges | ✅ |
| 2026-09-09 | 3-A | **INC-004**: "ollama judge" runs never reached the LLM (host can't resolve `ollama`); dishonest labeling fixed + `subject_answer` persisted + `evalforge rejudge` CLI + OllamaJudge hardening; suite 204 → 223 | ✅ fixed |
| 2026-09-09 | 3-B | **INC-004 RESOLVED**: `evalforge rejudge` executed in-network on the captured artifacts → `eval_report_alpha-agent_ollama-in-network_20260908T232826Z` (10/10) and `eval_report_smart-contract-rag_ollama-in-network_20260908T232921Z` (11/12, sr-001 skipped), 0 fallbacks; README + `LIVE_EVAL.md` show the final semantic numbers | ✅ resolved |

---

## 3. Incident reports

### INC-001 — Golden path: default dataset resolution broke for kebab-case subjects

- **When**: 2026-09-08, Phase 1 → 2 handoff — first CLI/e2e runs.
- **Symptom**: subjects are registered kebab-case (`alpha-agent`,
  `smart-contract-rag`) but the golden files are snake_case
  (`alpha_agent.json`, `smart_contract_rag.json`). The default path
  `data/golden/<subject>.json` resolved to a file that does not exist for
  either shipped subject, and the CLI and `run_e2e.py` each maintained their
  own path logic.
- **Impact**: the golden-path happy flow ("just run it") failed for the two
  subjects the project exists to evaluate; two divergent copies of the naming
  logic guaranteed future drift.
- **Root cause**: no single source of truth for the naming convention; the
  kebab→snake mapping was implicit and duplicated.
- **Fix** (commit `21f6ecb`): a shared pure helper,
  `evalforge.dataset.default_golden_filename(subject)` (kebab → snake), used by
  both the CLI (`_default_golden`) and `scripts/run_e2e.py`; `--golden` still
  overrides.
- **Verification**: 7 new deterministic tests (`TestDefaultGoldenFilename` +
  CLI kebab-resolution tests): suite 197 → **204**, all green, hermetic.
- **Lesson**: a naming convention must live in exactly one pure function that
  every entry point calls — never in two ad-hoc copies.

---

### INC-002 — Live e2e: subject timeout under shared-Ollama load (operational)

- **When**: 2026-09-08, Phase 3 — e2e run #4 (`smart-contract-rag`,
  `--judge ollama`, 22:05Z).
- **Symptom**: case `sr-001` recorded
  `subject error: smart-contract-rag timed out after 120.0s answering: 'What
  is a reentrancy attack and why is it dangerous in DeFi?'`.
- **Impact**: one of 12 cases could not be answered; the run completed with
  the remaining 11 cases and the honest overall verdict (FAIL).
- **Root cause**: shared local Ollama daemon under load (the same daemon
  serves the subjects' inference and other host workloads); the subject
  exceeded `CliSubject`'s default per-question timeout (120.0s). Operational
  resource contention — not a code bug.
- **Fix**: none in code — the designed subject-error contract handled it: the
  case was marked refused + `error` populated, the run continued, and the
  metrics were not polluted (hallucination 0.0000, no fabricated content).
  Tuning zone documented: per-subject timeouts (`CliSubject(timeout=...)`) and
  scheduling e2e when Ollama is idle.
- **Verification**: the raw artifact of that first-session run
  (`eval_report_smart-contract-rag_ollama_20260908T220529Z.md`) showed the
  error cell, 11/12 cases processed, exit code 1. The pattern **reproduced**
  in the current session — `sr-001` errored again in
  `eval_report_smart-contract-rag_heuristic_20260908T232709Z` (per-case error
  cell, `docs/LIVE_EVAL.md` §4) and was honestly **excluded** from the
  semantic rejudge (`n_skipped=1`, `docs/LIVE_EVAL.md` §5.2/§7.1).
- **Lesson**: a slow or crashed case must never abort an eval nor inject
  fabricated content — mark it, treat it as a refusal, continue, and audit it
  through the per-case `error` column.

---

### INC-003 — Heuristic faithfulness is harsh on free text → dual judge (design)

- **When**: 2026-09-08, Phase 3 — after the first real numbers landed.
- **Symptom**: `alpha-agent` faithfulness scored 0.0860 (heuristic run) and
  0.0877 (second run under the ollama judge label) — a lexical 0.08-ish floor
  on every run regardless of answer quality.
- **Root cause**: heuristic faithfulness measures containment of the golden
  reference context (keywords + doc ids + cited sources) in the answer. An
  agent that composes tool output into fluent prose will rarely contain the
  exact golden tokens — this is a property of the proxy for free-text
  subjects, not a hidden regression of the subject.
- **Decision**: no code change — documented as design. evalforge ships a
  **dual judge**: the deterministic heuristic is the CI-safe gate (zero
  network, reproducible), the `OllamaJudge` is the semantic route (0–5,
  structured, with a fallback that keeps deterministic scores and records the
  reason on failure). See `docs/ARCHITECTURE.md` §3/§5 and
  `docs/LIVE_EVAL.md` §9.
- **Verification**: all 4 e2e runs returned the same verdict (FAIL); the
  per-topic signals were stable across judge labels (note: the two "ollama"
  runs of that session actually used the deterministic fallback — see
  INC-004). README and `docs/LIVE_EVAL.md` document the distinction
  explicitly.
- **Lesson**: never gate quality on lexical faithfulness alone for free-text
  subjects — keep a deterministic gate for CI and use the LLM judge as the
  semantic layer.

---

### INC-004 — "ollama judge" runs never reached the LLM (host ≠ docker network)

- **Status**: **RESOLVED** (2026-09-09 — ciclo B, in-network rejudge executed).
- **When**: 2026-09-09 — review of the 2026-09-08 e2e artifacts by
  portfolio-reviewer.
- **Symptom**: artifacts 3–4 carry `"judge": "ollama"` while *every* per-case
  `judge_reason` is a `ConnectionError ... NameResolutionError('ollama', 11434)`.
  The README and `LIVE_EVAL.md` presented the aggregate tables of those runs
  without qualifying that no semantic judgment happened — and an unqualified
  "LLM-verified numbers" claim in a public portfolio is a false claim. The
  artifacts also lacked the subject's captured answer, so they could never be
  re-scored later.
- **Impact**: dishonest labeling of a portfolio deliverable; no path to a real
  semantic evaluation without re-running the slow, stochastic subjects.
- **Root cause**: `scripts/run_e2e.py` executes on the host (WSL), where
  `http://ollama:11434` does not resolve — the name exists only inside the
  `docker_default` network. The documented fallback (keep deterministic
  scores, record `judge_reason`) did its job *correctly*, but the run labels
  and docs did not surface it, and nothing persisted the subject responses for
  a later in-network re-score.
- **Fix** (4 parts, all verified by hermetic tests):
  1. **Honest labeling** — README / `LIVE_EVAL.md` / `ARCHITECTURE.md` /
     `DEVELOPMENT_LOG.md` now state exactly what those two reports contain
     (deterministic gate over that run's answers); the aggregate table is
     retitled "all four runs" with the caveat inline.
  2. **Answer persistence** — `runner`/`report` now capture
     `subject_answer`, `subject_sources`, `subject_retrieved_doc_ids` and the
     golden reference in every artifact (`tests/test_runner.py`,
     `TestAnswerPersistence`, 3 tests).
  3. **`evalforge rejudge`** — CLI subcommand that re-scores a captured
     artifact inside `docker_default` and writes
     `eval_report_<subject>_ollama-in-network_<ts>.{md,json}` next to the
     source (never overwrites; skipped cases are counted with reasons; case
     count preserved). 12 hermetic tests in `tests/test_rejudge.py`.
  4. **OllamaJudge hardening** — default timeout 120s → 180s, `keep_alive:
     "5m"` on every scoring payload, optional `warm_up()` (never raises);
     5 new hermetic tests in `tests/test_judges.py`.
  `scripts/run_e2e.py` also warns on stderr when `--judge ollama` runs on the
  host.
- **Verification**: suite 204 → **223** passing (`python3 -m pytest -q`,
  0.38s, hermetic). Docs updated to the honest reading; grep shows no numbers
  presented as semantic that are not.
- **Lesson**: probe judge connectivity *before* the e2e and label every number
  with what actually produced it. A fallback is only a feature if the report
  says the fallback ran — otherwise it is a silent lie in a public artifact.

- **Closure (2026-09-09, ciclo B)** — evidence that the incident is fully
  resolved, i.e. the semantic judge ran for real:
  1. `data/e2e/eval_report_alpha-agent_ollama-in-network_20260908T232826Z.{md,json}`
     — rejudge of
     `data/e2e/eval_report_alpha-agent_heuristic_20260908T230456Z.json`,
     `n_rejudged=10`, `n_skipped=0`, `skipped_reasons=[]`; semantic
     faithfulness 1.0000, answer_relevance 0.7500, hallucination_rate 0.5000.
  2. `data/e2e/eval_report_smart-contract-rag_ollama-in-network_20260908T232921Z.{md,json}`
     — rejudge of
     `data/e2e/eval_report_smart-contract-rag_heuristic_20260908T232709Z.json`,
     `n_rejudged=11`, `n_skipped=1`
     (`sr-001: original run errored (no subject response to re-score)` —
     honest exclusion, the case timed out in the original run); semantic
     faithfulness 0.9200, answer_relevance 0.9200, retrieval/citation values
     copied verbatim (0.0729 / 0.2500 / 0.0000 per the rejudge contract).
  3. **0 fallbacks**: no re-judged case carries a `ConnectionError` or a
     `lexical (token overlap)` reason — the only such row is the copied
     `sr-001` skip. Every semantic number in `LIVE_EVAL.md` §5 comes from a
     live LLM judgment.
  4. README and `docs/LIVE_EVAL.md` updated: dual-judge table subject×judge
     with the final numbers, no "pending" notes about semantic scores.
  5. Suite unchanged at **223 passing** (no code changes in ciclo B — docs
     only).

---

## 4. Security & privacy notes

| Item | Decision | Why |
|------|----------|-----|
| **Git author identity** | `eLSeR17 <112463505+eLSeR17@users.noreply.github.com>` | Public repo; no personal e-mail on GitHub. |
| **Secrets** | None — `.env` only via `.env.example` placeholders; `.gitignore` excludes `.env`, `data/reports/`, `data/e2e/` | Public repo rule: never ship keys, tokens, credentials or run artifacts with machine-local metadata. |
| **Confidentiality** | No internal infrastructure details (host paths, private service names, private ports) appear in repo content | The repo documents only its own stack (Docker network, Ollama service name), nothing from the host environment. |
| **Local-only LLM** | All inference via local Ollama; no cloud provider, no API keys | Cheaper, private, reproducible. |

## 5. How to reproduce the key checks

```bash
# Full deterministic suite (no Ollama, no network needed)
pip install -e .[dev]
pytest -q                                    # 223 passed

# Real e2e against the sibling containers (Docker + Ollama required)
PYTHONPATH=src python3 scripts/run_e2e.py --subject all --judge heuristic
PYTHONPATH=src python3 scripts/run_e2e.py --subject all --judge ollama

# In-network semantic re-scoring of a captured artifact (Docker network only)
docker run --rm --network docker_default \
  -v "$PWD":/app -w /app python:3.12-slim \
  sh -c 'pip install -q requests && PYTHONPATH=src python3 -m evalforge.cli rejudge \
    --artifact data/e2e/<artifact>.json --judge ollama \
    --golden data/golden/<subject>.json --model qwen2.5-coder:7b'
```

See `docs/LIVE_EVAL.md` for the canonical e2e record and full reproduction.