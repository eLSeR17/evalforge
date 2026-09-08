"""evalforge — a standalone, reusable LLM evaluation toolkit.

evalforge evaluates LLM pipelines and agents against curated *golden datasets*
using a dual-judge design:

- :class:`~evalforge.judges.HeuristicJudge` — a deterministic, dependency-free
  token-overlap judge that runs in CI with zero network access.
- :class:`~evalforge.judges.OllamaJudge` — an LLM-as-judge that calls a local
  Ollama instance and returns structured 0-5 scores.

A regression guard collapses per-case results into a single PASS / WARN / FAIL
verdict with a semantic exit code, so CI can gate on quality regressions.

The toolkit is intentionally *subject-agnostic*: any Python object exposing
:meth:`~evalforge.subjects.Subject.ask` can be evaluated. The included
``scripts/run_e2e.py`` demonstrates this by evaluating the two sibling
portfolio projects (alpha-agent and smart-contract-rag) from *outside*,
through ``docker exec`` — no code copied, no internals imported.
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]