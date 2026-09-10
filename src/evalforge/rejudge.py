"""In-network semantic re-scoring of a previous run's artifact.

The host running the e2e lives *outside* the ``docker_default`` network, so
``http://ollama:11434`` does not resolve there (``NameResolutionError`` — see
INC-004 in ``docs/DEVELOPMENT_LOG.md``). The ``OllamaJudge`` however **can**
reach Ollama from any container attached to ``docker_default``.

``rejudge_artifact`` closes that gap without re-running the subject: it loads
an existing report artifact (whose cases now persist the raw subject answer
and the golden contract), reconstructs ``(question, reference, answer)`` per
case, scores each one with the chosen judge, and writes a **new** artifact
(``..._ollama-in-network_<ts>.{md,json}``) — the original is never touched.

Honesty contract (same anti-cruelty rules as the runner):

- a case without a captured ``subject_answer`` is **kept verbatim**
  (``n_skipped`` is recorded in the metadata — we never invent answers);
- a case that errored during the original run is **kept verbatim** (an errored
  case has no subject response to re-score);
- a judge that fails (network / parse / timeout) keeps the deterministic
  fallback scores and records the reason in ``judge_reason`` — never a crash,
  never a silent zero;
- the number of cases is always preserved.

The golden dataset is the authoritative source for the case contract when
``golden_cases`` is provided; otherwise the contract persisted in the artifact
itself (``expected_keywords`` / ``doc_ids`` / ``refuse`` /
``golden_reference``) is used — artifacts produced by the current runner are
self-contained.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime

from .judges import Judge, PerCaseJudge
from .metrics import aggregate
from .models import EvalCase, EvalReport, PerCaseResult, SubjectAnswer
from .runner import (
    EXIT_CODE_FAIL,
    EXIT_CODE_PASS,
    RegressionThresholds,
    Verdict,
    _lexically_refused,
    _verdict,
)

#: JSON keys of a serialised PerCaseResult case (see models.PerCaseResult).
_CASE_KEYS: tuple[str, ...] = (
    "case_id",
    "topic",
    "question",
    "answered",
    "refused",
    "faithfulness",
    "answer_relevance",
    "citation_accuracy",
    "context_precision",
    "context_recall",
    "correct_refusal",
    "hallucination",
    "error",
    "judge_reason",
    "subject_answer",
    "subject_refused",
    "subject_sources",
    "subject_retrieved_doc_ids",
    "golden_reference",
    "expected_keywords",
    "doc_ids",
    "refuse",
)


@dataclass(frozen=True)
class RejudgeOutcome:
    """Result of re-scoring an artifact: the new report plus honest counts."""

    report: EvalReport
    n_rejudged: int
    n_skipped: int
    skipped_reasons: list[str] = field(default_factory=list)


def _skip_reason(case: dict, golden_cases: Mapping[str, EvalCase] | None) -> str | None:
    """Why this case cannot be re-scored, or ``None`` when it can."""
    if "subject_answer" not in case:
        return "no subject_answer captured in artifact"
    if case.get("error"):
        return "original run errored (no subject response to re-score)"
    if golden_cases is not None:
        if case["case_id"] not in golden_cases:
            return "case id not found in the provided golden dataset"
        return None
    if not (case.get("expected_keywords") or case.get("doc_ids")) and not case.get(
        "golden_reference"
    ):
        return "artifact lacks the golden contract (expected_keywords/doc_ids/golden_reference)"
    return None


def _case_from_dict(case: dict) -> PerCaseResult:
    """Rebuild a :class:`PerCaseResult` from a serialised case dict.

    Tolerant by design: missing snapshot keys (old artifacts) fall back to
    empty values so the case is preserved verbatim for the scored fields.
    """
    return PerCaseResult(
        case_id=str(case.get("case_id", "")),
        topic=str(case.get("topic", "")),
        question=str(case.get("question", "")),
        answered=bool(case.get("answered", False)),
        refused=bool(case.get("refused", False)),
        faith=_opt_float(case.get("faithfulness")),
        rel=_opt_float(case.get("answer_relevance")),
        cit=_opt_float(case.get("citation_accuracy")),
        p_at_k=_opt_float(case.get("context_precision")),
        rec=_opt_float(case.get("context_recall")),
        refusal_correct=bool(case.get("correct_refusal", False)),
        hallu=_opt_opt_bool(case.get("hallucination")),
        error=str(case.get("error", "")),
        judge_reason=str(case.get("judge_reason", "")),
        subject_answer=str(case.get("subject_answer", "")),
        subject_refused=bool(case.get("subject_refused", False)),
        subject_sources=list(case.get("subject_sources", []) or []),
        subject_retrieved_doc_ids=list(case.get("subject_retrieved_doc_ids", []) or []),
        golden_reference=str(case.get("golden_reference", "")),
        expected_keywords=list(case.get("expected_keywords", []) or []),
        doc_ids=list(case.get("doc_ids", []) or []),
        refuse=bool(case.get("refuse", False)),
    )


def _rebuild_eval_case(
    case: dict, golden_cases: Mapping[str, EvalCase] | None
) -> EvalCase:
    """Rebuild the golden :class:`EvalCase` for a serialised case.

    The golden dataset (when provided) is authoritative; otherwise the
    contract persisted in the artifact is used.
    """
    if golden_cases is not None and case["case_id"] in golden_cases:
        return golden_cases[case["case_id"]]
    return EvalCase(
        id=str(case.get("case_id", "")),
        topic=str(case.get("topic", "")),
        question=str(case.get("question", "")),
        expected_keywords=list(case.get("expected_keywords", []) or []),
        refuse=bool(case.get("refuse", False)),
        doc_ids=list(case.get("doc_ids", []) or []),
    )


def _rebuild_subject_answer(case: dict) -> SubjectAnswer:
    """Rebuild the recorded :class:`SubjectAnswer` from the artifact."""
    return SubjectAnswer(
        answer=str(case.get("subject_answer", "")),
        sources=list(case.get("subject_sources", []) or []),
        retrieved_doc_ids=list(case.get("subject_retrieved_doc_ids", []) or []),
        refused=bool(case.get("subject_refused", False)),
    )


def rejudge_artifact(
    artifact: dict,
    *,
    judge: Judge,
    thresholds: RegressionThresholds | None = None,
    golden_cases: Mapping[str, EvalCase] | None = None,
    metadata: dict | None = None,
    artifact_path: str = "",
) -> RejudgeOutcome:
    """Re-score every re-judgeable case of *artifact* with *judge*.

    ``artifact`` is a parsed evalforge report JSON (``EvalReport.to_dict``).
    Returns a :class:`RejudgeOutcome` carrying the new
    :class:`~evalforge.models.EvalReport` (same subject, same case count)
    plus the honest ``n_rejudged`` / ``n_skipped`` bookkeeping.

    Cases are skipped verbatim when they have no ``subject_answer`` or when
    the original run errored. Judge failures fall back to the deterministic
    scores with the reason recorded in ``judge_reason`` — never a crash,
    never a silent zero.
    """
    raw_cases = artifact.get("cases", [])
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("artifact JSON has no 'cases' list — not an evalforge report")

    thr = thresholds or _thresholds_from_artifact(artifact)
    skipped_reasons: list[str] = []
    case_results: list[PerCaseResult] = []
    rebuilt_cases: list[EvalCase] = []
    n_rejudged = 0

    for raw in raw_cases:
        # Rebuild the golden contract for *every* case (skipped ones keep
        # their original scores but still need the contract for aggregation).
        rebuilt_cases.append(_rebuild_eval_case(raw, golden_cases))
        reason = _skip_reason(raw, golden_cases)
        if reason is not None:
            n_skipped_note = f"{raw.get('case_id', '?')}: {reason}"
            skipped_reasons.append(n_skipped_note)
            case_results.append(_case_from_dict(raw))  # verbatim
            continue

        case = _rebuild_eval_case(raw, golden_cases)
        answer = _rebuild_subject_answer(raw)
        scored: PerCaseJudge = judge.score(case, answer)
        refused = answer.refused or _lexically_refused(answer.answer)
        case_results.append(
            PerCaseResult(
                case_id=case.id,
                topic=case.topic,
                question=case.question,
                answered=not refused,
                refused=refused,
                faith=scored.faith,
                rel=scored.rel,
                cit=scored.cit,
                p_at_k=scored.p_at_k,
                rec=scored.rec,
                refusal_correct=bool(scored.refusal_correct),
                hallu=scored.hallu,
                error="",
                judge_reason=scored.reasoning or scored.error,
                subject_answer=answer.answer,
                subject_refused=answer.refused,
                subject_sources=list(answer.sources),
                subject_retrieved_doc_ids=list(answer.retrieved_doc_ids),
                golden_reference=case.reference_text(),
                expected_keywords=list(case.expected_keywords),
                doc_ids=list(case.doc_ids),
                refuse=case.refuse,
            )
        )
        n_rejudged += 1

    metrics = aggregate(
        [
            {
                "answered": c.answered,
                "refused": c.refused,
                "faith": c.faith,
                "rel": c.rel,
                "cit": c.cit,
                "p_at_k": c.p_at_k,
                "rec": c.rec,
                "refusal_correct": c.refusal_correct,
                "hallu": c.hallu,
                "expected_refusal": ec.refuse,
            }
            for c, ec in zip(case_results, rebuilt_cases)
        ]
    )
    verdict = _verdict(metrics, thr)

    run_metadata: dict = {
        "rejudge_of": artifact_path,
        "original_judge": artifact.get("judge", "?"),
        "n_rejudged": n_rejudged,
        "n_skipped": len(skipped_reasons),
        "skipped_reasons": skipped_reasons,
    }
    if metadata:
        run_metadata.update(metadata)

    report = EvalReport(
        subject=str(artifact.get("subject", "?")),
        judge=judge.name,
        status=verdict.value,
        metrics=metrics,
        cases=case_results,
        thresholds=thr.as_dict(),
        metadata=run_metadata,
        exit_code=EXIT_CODE_PASS if verdict is not Verdict.FAIL else EXIT_CODE_FAIL,
        created_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    return RejudgeOutcome(
        report=report,
        n_rejudged=n_rejudged,
        n_skipped=len(skipped_reasons),
        skipped_reasons=skipped_reasons,
    )


def _thresholds_from_artifact(artifact: Mapping) -> RegressionThresholds:
    """Rebuild the regression thresholds recorded in the artifact."""
    raw_thresholds = artifact.get("thresholds") or {}
    known = {
        k: v
        for k, v in raw_thresholds.items()
        if k in RegressionThresholds.__dataclass_fields__
    }
    return RegressionThresholds(**known)


def _opt_float(value) -> float | None:
    """Coerce a JSON number-or-null to ``float | None``."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _opt_opt_bool(value) -> bool | None:
    """Coerce a JSON bool-or-null to ``bool | None`` (hallucination)."""
    if value is None:
        return None
    return bool(value)


__all__ = [
    "RejudgeOutcome",
    "rejudge_artifact",
]