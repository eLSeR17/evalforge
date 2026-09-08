"""Eval orchestration and regression guard.

``run_eval`` runs a list of golden cases through a :class:`Subject`, scores
every answer with a :class:`Judge`, aggregates the per-case signals into the
eight report metrics, and collapses them into a verdict — PASS / WARN / FAIL —
against a set of :class:`RegressionThresholds`.

The thresholds are the **regression guard**: CI fails (FAIL) when any
measurable metric degrades past its floor/cap. The banding logic mirrors the
sibling ``smart-contract-rag`` runner (``evals/runner.py::_verdict``) with one
documented generalisation: metrics that could not be measured for *any* case
(``None`` aggregate) are excluded from the verdict — they show as ``n/a`` in
the report instead of forcing a vacuous pass or fail.

Design notes
------------
- The subject is injected, so this module works with a real container-backed
  subject or any deterministic fake. The unit tests never touch the network,
  Docker, or Ollama.
- Execution is strictly **sequential** (no threads) — the ecosystem rule
  against threads applies to the produced code as well.
- A refusal is the union of the adapter's mark (``SubjectAnswer.refused``)
  and the lexical signal; empty answers count as refusals.
- A subject that raises (timeout, crash) marks that case as errored instead
  of aborting the whole run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Protocol, Sequence

from .judges import HeuristicJudge, Judge
from .metrics import aggregate
from .models import EvalCase, EvalReport, PerCaseResult, SubjectAnswer
from .subjects import Subject, SubjectError


class Verdict(str, Enum):
    """Overall verdict of an eval run against the regression thresholds."""

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


#: Semantic exit codes. WARN is *non-fatal* in CI by design (review-and-decide):
#: see ARCHITECTURE.md § "Exit codes and CI".
EXIT_CODE_PASS = 0
EXIT_CODE_FAIL = 1


@dataclass(frozen=True)
class RegressionThresholds:
    """Quality floors/caps for the regression guard.

    Higher is better for the ``min_*`` floors; lower is better for
    ``max_hallucination_rate``. ``warn_margin`` (default 0.05) defines the
    band below/above a guard where the run reports WARN instead of FAIL.
    Only non-``None`` aggregated metrics are guarded.
    """

    max_hallucination_rate: float = 0.10
    min_faithfulness: float = 0.70
    min_answer_relevance: float = 0.60
    min_citation_accuracy: float = 0.60
    min_context_precision: float = 0.50
    min_context_recall: float = 0.50
    min_answer_rate: float = 0.70
    warn_margin: float = 0.05

    def as_dict(self) -> dict[str, float]:
        return {
            "max_hallucination_rate": self.max_hallucination_rate,
            "min_faithfulness": self.min_faithfulness,
            "min_answer_relevance": self.min_answer_relevance,
            "min_citation_accuracy": self.min_citation_accuracy,
            "min_context_precision": self.min_context_precision,
            "min_context_recall": self.min_context_recall,
            "min_answer_rate": self.min_answer_rate,
            "warn_margin": self.warn_margin,
        }

    def with_overrides(self, overrides: dict[str, float] | None) -> "RegressionThresholds":
        """Return a copy with the given threshold key overrides applied."""
        if not overrides:
            return self
        unknown = set(overrides) - set(self.as_dict())
        if unknown:
            raise ValueError(f"unknown threshold keys: {sorted(unknown)}")
        values = self.as_dict()
        values.update(overrides)
        return RegressionThresholds(**values)


@dataclass
class RunConfig:
    """Free-form metadata captured for every run."""

    dataset_source: str = ""
    n_cases: int = 0
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        out = {"dataset": self.dataset_source, "n_cases": self.n_cases}
        out.update(self.extra)
        return out


def run_eval(
    subject: Subject,
    cases: Sequence[EvalCase],
    *,
    judge: Judge | None = None,
    thresholds: RegressionThresholds | None = None,
    metadata: dict | None = None,
    dataset_source: str = "",
) -> EvalReport:
    """Evaluate *subject* against *cases* and produce an :class:`EvalReport`.

    Guards:
    - an empty case list raises ``ValueError`` (a guard with no data is a
      configuration error, not a pass);
    - an empty/stale subject answer is always mapped to a refusal;
    - a subject exception marks the case as errored (refused=True);
    - metrics that cannot be measured are ``None`` and excluded.
    """
    if not cases:
        raise ValueError("run_eval requires at least one golden case")

    thr = thresholds or RegressionThresholds()
    active_judge: Judge = judge or HeuristicJudge()

    case_results: list[PerCaseResult] = []
    for case in cases:
        try:
            answer = subject.ask(case.question)
        except SubjectError as exc:
            answer = SubjectAnswer(answer="", refused=True)
            error = f"subject error: {exc}"
        except Exception as exc:  # unexpected adapter failure
            answer = SubjectAnswer(answer="", refused=True)
            error = f"subject error: {type(exc).__name__}: {exc}"
        else:
            error = ""

        scored = active_judge.score(case, answer)
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
                error=error,
                judge_reason=scored.reasoning or scored.error,
            )
        )

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
                "expected_refusal": case.refuse,
            }
            for c, case in zip(case_results, cases)
        ]
    )
    verdict = _verdict(metrics, thr)

    run_metadata = RunConfig(
        dataset_source=dataset_source, n_cases=len(cases)
    ).as_dict()
    if metadata:
        run_metadata.update(metadata)

    return EvalReport(
        subject=subject.name,
        judge=active_judge.name,
        status=verdict.value,
        metrics=metrics,
        cases=case_results,
        thresholds=thr.as_dict(),
        metadata=run_metadata,
        exit_code=EXIT_CODE_PASS if verdict is not Verdict.FAIL else EXIT_CODE_FAIL,
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def _lexically_refused(answer: str) -> bool:
    from .metrics import is_refusal

    return is_refusal(answer)


def _verdict(metrics: dict[str, float | None], thr: RegressionThresholds) -> Verdict:
    """Compute PASS/WARN/FAIL from the aggregated metrics and thresholds.

    FAIL if any *measurable* metric breaches its guard. WARN if any metric is
    within the warn-margin band of its guard (approaching but not past).
    Otherwise PASS. ``None`` aggregates are skipped (not measurable).
    """
    failed: list[str] = []
    warned: list[str] = []

    # Rate metric: lower is better.
    rate = metrics.get("hallucination_rate")
    guard = thr.max_hallucination_rate
    if rate is not None:
        band_low = guard - thr.warn_margin
        if rate > guard:
            failed.append("hallucination_rate")
        elif guard < 1.0 and band_low > 0 and rate > band_low:
            warned.append("hallucination_rate")

    # Floor metrics: higher is better.
    floors: list[tuple[str, float, float]] = [
        ("faithfulness", "min_faithfulness", thr.min_faithfulness),
        ("answer_relevance", "min_answer_relevance", thr.min_answer_relevance),
        ("citation_accuracy", "min_citation_accuracy", thr.min_citation_accuracy),
        ("context_precision", "min_context_precision", thr.min_context_precision),
        ("context_recall", "min_context_recall", thr.min_context_recall),
        ("answer_rate", "min_answer_rate", thr.min_answer_rate),
    ]
    for label, _key, floor in floors:
        value = metrics.get(label)
        if value is None:
            continue
        band_high = floor + thr.warn_margin
        if value < floor:
            failed.append(label)
        elif floor > 0 and band_high < 1 and value < band_high:
            warned.append(label)

    if failed:
        return Verdict.FAIL
    if warned:
        return Verdict.WARN
    return Verdict.PASS


__all__ = [
    "EXIT_CODE_FAIL",
    "EXIT_CODE_PASS",
    "RegressionThresholds",
    "RunConfig",
    "Verdict",
    "run_eval",
]