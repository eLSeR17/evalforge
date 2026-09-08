"""Judges: deterministic heuristic and LLM-as-judge.

A *judge* scores one :class:`~evalforge.models.SubjectAnswer` against one
:class:`~evalforge.models.EvalCase`, producing a :class:`PerCaseJudge` with
the seven per-case signals:

- ``faith`` — faithfulness/grounding (``[0,1]`` or ``None``)
- ``rel`` — answer relevance (``[0,1]``)
- ``cit`` — citation accuracy (``[0,1]`` or ``None``)
- ``p_at_k`` — context precision (``[0,1]`` or ``None``)
- ``rec`` — context recall (``[0,1]`` or ``None``)
- ``refusal_correct`` — was the refusal decision right?
- ``hallu`` — fabrication signal (``bool`` or ``None``)

Two implementations are provided, mirroring the dual-judge design of the
sibling ``smart-contract-rag`` project:

- :class:`HeuristicJudge` — deterministic, token-overlap based, runs anywhere
  with zero network. This is the default (CI-safe).
- :class:`OllamaJudge` — LLM-as-judge on a 0-5 scale with a structured JSON
  contract, calling ``{OLLAMA_URL}/api/chat`` inside the docker network. On
  any failure it *falls back to the deterministic scores* and records the
  error in the result — it never raises.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Protocol

import requests

from .metrics import (
    answer_relevance,
    citation_accuracy,
    context_precision,
    context_recall,
    correct_refusal,
    extract_doc_id,
    faithfulness,
    hallucination,
    is_refusal,
    tokens,
)
from .models import EvalCase, SubjectAnswer

#: Upper bound of the LLM judge scale. Heuristic scores are already in
#: ``[0,1]``; LLM scores are normalized ``judge_score / 5`` downstream.
JUDGE_MAX_SCORE: float = 5.0

#: Default Ollama endpoint inside the docker_default network. Never use
#: ``localhost`` from a container — it will not resolve.
DEFAULT_OLLAMA_URL: str = "http://ollama:11434"
DEFAULT_OLLAMA_MODEL: str = "qwen2.5:7b"

_OLLAMA_ENV_URL = "OLLAMA_URL"
_OLLAMA_ENV_MODEL = "OLLAMA_MODEL"


@dataclass(frozen=True)
class PerCaseJudge:
    """The judge's per-case output (all values in ``[0,1]`` where measurable)."""

    case_id: str
    faith: float | None = None
    rel: float | None = None
    cit: float | None = None
    p_at_k: float | None = None
    rec: float | None = None
    refusal_correct: bool | None = None
    hallu: bool | None = None
    reasoning: str = ""
    error: str = ""


class Judge(Protocol):
    """Any scorer of (case, answer) pairs."""

    name: str

    def score(self, case: EvalCase, answer: SubjectAnswer) -> PerCaseJudge:
        ...


def _refused(case: EvalCase, answer: SubjectAnswer) -> bool:
    """Refusal as marked by the adapter, re-verified lexically."""
    return answer.refused or is_refusal(answer.answer)


def _reference_context(case: EvalCase, answer: SubjectAnswer) -> str:
    """The golden reference context for faithfulness:
    expected keywords + declared doc ids + the sources the subject exposed."""
    parts = [*case.expected_keywords, *case.doc_ids, *answer.sources]
    return " ".join(parts)


def _has_evidence(answer: SubjectAnswer) -> bool:
    return bool(answer.sources or answer.retrieved_doc_ids)


def score_deterministic(case: EvalCase, answer: SubjectAnswer) -> PerCaseJudge:
    """Compute all seven per-case signals deterministically (shared by all
    judges; OllamaJudge only overrides faith/rel with the LLM scores)."""
    refused = _refused(case, answer)
    return PerCaseJudge(
        case_id=case.id,
        faith=faithfulness(answer.answer, ref_context=_reference_context(case, answer)),
        rel=answer_relevance(answer.answer, expected_keywords=case.expected_keywords),
        cit=citation_accuracy(answer.sources, case.doc_ids, refused=refused),
        p_at_k=context_precision(answer.retrieved_doc_ids, case.doc_ids),
        rec=context_recall(answer.retrieved_doc_ids, case.doc_ids),
        refusal_correct=correct_refusal(refused=refused, expected_refusal=case.refuse),
        hallu=hallucination(
            refused=refused,
            expected_refusal=case.refuse,
            has_evidence=_has_evidence(answer),
        ),
        reasoning="lexical (token overlap)",
    )


class HeuristicJudge:
    """Deterministic judge built from the lexical metrics (no network).

    Composes :func:`score_deterministic`. Fully reproducible — the default
    judge for CI and for the unit test suite.
    """

    name = "heuristic"

    def score(self, case: EvalCase, answer: SubjectAnswer) -> PerCaseJudge:
        return score_deterministic(case, answer)


# ---------------------------------------------------------------------------
# Ollama LLM-as-judge
# ---------------------------------------------------------------------------


class OllamaJudge:
    """LLM-as-judge that calls a local Ollama model for structured scoring.

    The model is asked to score *faithfulness* and *relevance* on a 0-5 scale
    and reply with a single JSON object::

        {"faithfulness": 4, "relevance": 3, "reasoning": "..."}

    The output is parsed with a defensive cascade (direct parse -> fenced
    block -> brace matching -> trailing-comma fix -> double-brace peel), so a
    slightly malformed model response never crashes the eval. On any failure
    the deterministic scores are kept and ``error`` is populated.

    Every HTTP attempt creates a *fresh* request (never a reused one) — the
    retry lesson learned on the sibling evaluation pipelines applies here too.
    """

    name = "ollama"

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
        max_attempts: int = 2,
        http_post: Callable[..., requests.Response] | None = None,
    ) -> None:
        import os

        if not base_url:
            base_url = os.environ.get(_OLLAMA_ENV_URL) or DEFAULT_OLLAMA_URL
        if not model:
            model = os.environ.get(_OLLAMA_ENV_MODEL) or DEFAULT_OLLAMA_MODEL
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_attempts = max(max_attempts, 1)
        #: Injection point for the unit tests (a fake POST keeps tests hermetic).
        self._post = http_post or requests.post

    # ------------------------------------------------------------------
    def score(self, case: EvalCase, answer: SubjectAnswer) -> PerCaseJudge:
        base = score_deterministic(case, answer)
        faith, rel, reasoning, error = self._call_llm(case, answer)
        if error:
            return PerCaseJudge(
                case_id=case.id,
                faith=base.faith,
                rel=base.rel,
                cit=base.cit,
                p_at_k=base.p_at_k,
                rec=base.rec,
                refusal_correct=base.refusal_correct,
                hallu=base.hallu,
                reasoning="",
                error=error,
            )
        return PerCaseJudge(
            case_id=case.id,
            faith=faith,
            rel=rel,
            cit=base.cit,
            p_at_k=base.p_at_k,
            rec=base.rec,
            refusal_correct=base.refusal_correct,
            hallu=base.hallu,
            reasoning=reasoning,
        )

    # ------------------------------------------------------------------
    def _call_llm(
        self, case: EvalCase, answer: SubjectAnswer
    ) -> tuple[float | None, float | None, str, str]:
        """Call the judge model; returns (faith, rel, reasoning, error)."""
        last_error = ""
        for _ in range(self.max_attempts):
            try:
                # Fresh payload per attempt: never reuse a mutated request body
                # (the retry lesson: reusing request objects masks real errors).
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": self._user_prompt(case, answer)},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 256},
                }
                response = self._post(
                    f"{self.base_url}/api/chat", json=payload, timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()
                content = (data.get("message") or {}).get("content", "") or ""
            except Exception as exc:  # network / HTTP / parse -> retry fresh
                last_error = f"{type(exc).__name__}: {exc}"
                continue

            parsed = _extract_json_object(content)
            if parsed is None:
                last_error = "judge output did not parse as JSON"
                continue
            faith_raw = parsed.get("faithfulness", parsed.get("faith"))
            rel_raw = parsed.get("relevance", parsed.get("rel"))
            if faith_raw is None or rel_raw is None:
                last_error = f"judge JSON missing faithfulness/relevance: {parsed!r}"
                continue
            reasoning = str(parsed.get("reasoning", "")).strip()
            return (
                _normalize(faith_raw),
                _normalize(rel_raw),
                reasoning,
                "",
            )
        return None, None, "", last_error

    # ------------------------------------------------------------------
    def _user_prompt(self, case: EvalCase, answer: SubjectAnswer) -> str:
        sources = "\n".join(f"- {s}" for s in answer.sources) or "(none provided)"
        doc_ids = ", ".join(case.doc_ids) or "(none declared)"
        keywords = ", ".join(case.expected_keywords) or "(none)"
        expected = "REFUSE (the answer should decline)" if case.refuse else "ANSWER (answerable)"
        refused_flag = "(the candidate refused)" if _refused(case, answer) else ""
        return (
            f"QUESTION:\n{case.question}\n\n"
            f"EXPECTED KEYWORDS:\n{keywords}\n\n"
            f"REFERENCE DOC IDS:\n{doc_ids}\n\n"
            f"CITED SOURCES:\n{sources}\n\n"
            f"EXPECTED BEHAVIOR: {expected}\n"
            f"CANDIDATE ANSWER {refused_flag}:\n{answer.answer or '(empty)'}\n"
        )


_JUDGE_SYSTEM_PROMPT = """\
You are a strict, impartial evaluation judge for an AI system. You will be
given a QUESTION, the golden EXPECTED KEYWORDS, the REFERENCE DOC IDS, the
CITED SOURCES the answer was built from, and the CANDIDATE ANSWER.

Score the candidate answer on TWO axes, each 0 to 5:
- faithfulness: is every claim in the candidate consistent with the golden
  reference context (keywords, reference documents, cited sources), with no
  invented facts? 5 = fully consistent/grounded, 0 = fabricated.
- relevance: does the candidate directly and correctly address the question?
  5 = perfect, 0 = off-topic or empty.

If the candidate is a refusal (says it cannot/unable/does not know):
faithfulness is usually high (it invented nothing) and relevance is low,
UNLESS a refusal was the expected behavior for this case (trap questions or
incomplete input), in which case a refusal is the CORRECT answer.

Respond with a single JSON object and nothing else:
{"faithfulness": <int or float 0-5>, "relevance": <int or float 0-5>, "reasoning": "<one short line>"}
"""


def _normalize(value, scale: float = JUDGE_MAX_SCORE) -> float:
    """Clamp a 0-5 judge score into [0, 1]."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(max(number / scale, 0.0), 1.0)


def _extract_json_object(text: str) -> dict | None:
    """Best-effort extraction of a JSON object from a model response.

    Cascade (from the sibling projects' lessons):
    1. fenced ```json ... ``` / ``` ... ``` blocks
    2. outermost brace substring (first ``{`` to last ``}``)
    3. double-brace peel (``{{ ... }}``)
    4. trailing-comma repair applied to every candidate
    5. the raw text itself
    """
    if not text or not text.strip():
        return None
    candidates: list[str] = []

    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        candidates.append(fence.group(1))

    if "{" in text and "}" in text:
        candidates.append(text[text.index("{"): text.rindex("}") + 1])

    stripped = text.strip()
    if stripped.startswith("{{") and stripped.endswith("}}"):
        candidates.append(stripped[1:-1])

    candidates.append(stripped)

    for candidate in candidates:
        for attempt in (candidate, _fix_trailing_commas(candidate)):
            try:
                data = json.loads(attempt)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(data, dict):
                return data
    return None


def _fix_trailing_commas(text: str) -> str:
    """Remove trailing commas before ``}`` / ``]`` (common LLM slip)."""
    return re.sub(r",\s*([}\]])", r"\1", text)


# Re-exported for tests/API convenience.
__all__ = [
    "DEFAULT_OLLAMA_URL",
    "DEFAULT_OLLAMA_MODEL",
    "HeuristicJudge",
    "Judge",
    "JUDGE_MAX_SCORE",
    "OllamaJudge",
    "PerCaseJudge",
    "extract_doc_id",
    "score_deterministic",
]