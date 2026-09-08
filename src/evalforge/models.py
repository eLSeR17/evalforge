"""Core value objects for the evalforge toolkit.

These dataclasses are deliberately dependency-free so that the whole eval
pipeline can be reasoned about, tested, and serialised without pulling in any
LLM, vector store, or external service.

The design mirrors the eval models used by the sibling
``smart-contract-rag`` project (``smart_contract_rag.evals``) but is
*generalised*: nothing here knows about RAG, Smart Contracts, finance, or any
specific domain.

Key objects
-----------
- :class:`EvalCase` — one golden sample (a question plus the expectation).
- :class:`SubjectAnswer` — what a subject returned for a question.
- :class:`PerCaseResult` — the scored outcome of one case.
- :class:`EvalReport` — the full run: metrics, verdict, cases, metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvalCase:
    """A single golden sample.

    Parameters
    ----------
    id:
        Unique identifier of the case within its dataset.
    topic:
        Free-form domain label used to group/analyse cases (e.g. ``"price"``,
        ``"reentrancy"``, ``"trap"``).
    question:
        The user query the subject must answer.
    expected_keywords:
        Substantive terms a *correct* answer is expected to contain. Used by
        the deterministic judge for lexical relevance. Must be non-empty for
        answerable cases and empty for refusal cases (validated on load).
    refuse:
        ``True`` for questions the subject should *decline* to answer: either
        a grounded refusal (incomplete input, out-of-scope) or a *trap*
        (question with no backing data — answering it is fabrication).
    doc_ids:
        Canonical identifiers of the source documents that back the answer.
        Used for citation accuracy and context precision/recall when the
        subject exposes retrieval provenance. Optional.
    """

    id: str
    topic: str
    question: str
    expected_keywords: list[str] = field(default_factory=list)
    refuse: bool = False
    doc_ids: list[str] = field(default_factory=list)

    def reference_text(self) -> str:
        """The golden reference context: expected keywords + declared doc ids.

        This is the *golden* evidence an answer is measured against. It is
        persisted per case in the run artifact (``golden_reference``) so that
        an artifact can be re-scored later (``rejudge``) without re-loading
        the original dataset: question + reference + answer are all captured.
        """
        return " ".join([*self.expected_keywords, *self.doc_ids]).strip()

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        flag = " (refuse)" if self.refuse else ""
        return f"{self.id}: {self.question}{flag}"


@dataclass(frozen=True)
class SubjectAnswer:
    """What a subject returned for a single question.

    Parameters
    ----------
    answer:
        The subject's final answer text. An empty answer is treated as a
        refusal (nothing was produced).
    sources:
        Human-readable source labels the answer cites or is grounded in
        (e.g. ``"aave-v3 (page 41)"``). Filled by the subject adapter when the
        subject exposes provenance.
    retrieved_doc_ids:
        Ordered list of document ids the subject retrieved, when the adapter
        can observe retrieval (e.g. from the RAG CLI output). Optional: when
        absent, context precision/recall cannot be measured.
    refused:
        ``True`` when the subject declined to answer (matched by the
        adapter's refusal pattern, a timeout, or an empty response). This is
        the adapter's *mark*; the runner re-verifies it lexically.
    """

    answer: str
    sources: list[str] = field(default_factory=list)
    retrieved_doc_ids: list[str] = field(default_factory=list)
    refused: bool = False

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        flag = " (refused)" if self.refused else ""
        return f"<SubjectAnswer: {len(self.answer)} chars{flag}, {len(self.sources)} sources>"


@dataclass(frozen=True)
class PerCaseResult:
    """The scored outcome of evaluating one golden case.

    All scalar metrics are in ``[0, 1]`` where measurable; ``None`` means the
    metric is *not applicable* to this case (e.g. citation accuracy when the
    case declares no ``doc_ids``). ``None`` values are excluded from the
    aggregate metrics — never silently treated as 0.

    The class also carries a **captured snapshot** of the run for later
    re-scoring: the raw ``subject_answer`` text, the refusal mark, the exposed
    sources/retrieval, and the golden contract (``golden_reference``,
    ``expected_keywords``, ``doc_ids``, ``refuse``). ``evalforge rejudge``
    uses this snapshot to reconstruct (question, reference, answer) and score
    the same responses with a different judge — no subject re-run needed.
    """

    case_id: str
    topic: str
    question: str
    answered: bool
    refused: bool
    faith: float | None
    rel: float | None
    cit: float | None
    p_at_k: float | None
    rec: float | None
    refusal_correct: bool
    hallu: bool | None
    error: str = ""
    judge_reason: str = ""
    # -- Captured snapshot for re-scoring (rejudge) -------------------------
    # The raw subject response and the golden contract are persisted in the
    # JSON artifact so a later run can reconstruct (question, reference,
    # answer) and re-score with a different judge without re-running the
    # subject. ``subject_sources`` / ``subject_retrieved_doc_ids`` keep the
    # deterministic fallback reproducible; ``expected_keywords`` / ``doc_ids``
    # / ``refuse`` / ``golden_reference`` rebuild the golden case.
    subject_answer: str = ""
    subject_refused: bool = False
    subject_sources: list[str] = field(default_factory=list)
    subject_retrieved_doc_ids: list[str] = field(default_factory=list)
    golden_reference: str = ""
    expected_keywords: list[str] = field(default_factory=list)
    doc_ids: list[str] = field(default_factory=list)
    refuse: bool = False

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "topic": self.topic,
            "question": self.question,
            "answered": self.answered,
            "refused": self.refused,
            "faithfulness": _round_opt(self.faith),
            "answer_relevance": _round_opt(self.rel),
            "citation_accuracy": _round_opt(self.cit),
            "context_precision": _round_opt(self.p_at_k),
            "context_recall": _round_opt(self.rec),
            "correct_refusal": self.refusal_correct,
            "hallucination": self.hallu,
            "error": self.error,
            "judge_reason": self.judge_reason,
            "subject_answer": self.subject_answer,
            "subject_refused": self.subject_refused,
            "subject_sources": list(self.subject_sources),
            "subject_retrieved_doc_ids": list(self.subject_retrieved_doc_ids),
            "golden_reference": self.golden_reference,
            "expected_keywords": list(self.expected_keywords),
            "doc_ids": list(self.doc_ids),
            "refuse": self.refuse,
        }


@dataclass
class EvalReport:
    """The aggregated result of one eval run.

    Parameters
    ----------
    subject:
        Name of the evaluated subject.
    judge:
        Name of the judge used (``"heuristic"`` or ``"ollama"``).
    status:
        Overall verdict: PASS / WARN / FAIL.
    metrics:
        Aggregate metrics keyed by name (faithfulness, answer_relevance,
        citation_accuracy, context_precision, context_recall, answer_rate,
        correct_refusal_rate, hallucination_rate). A value of ``None`` means
        the metric could not be measured for any case and was excluded from
        the verdict.
    cases:
        Ordered per-case results.
    thresholds:
        The regression thresholds used for the verdict.
    metadata:
        Free-form run metadata (dataset path, model, timestamps, ...).
    exit_code:
        Semantic exit code: ``0`` PASS (WARN is non-fatal by design),
        ``1`` FAIL. See ``docs/ARCHITECTURE.md`` for the rationale.
    created_at:
        ISO-8601 UTC timestamp of the run.
    """

    subject: str
    judge: str
    status: str = "PASS"
    metrics: dict[str, float | None] = field(default_factory=dict)
    cases: list[PerCaseResult] = field(default_factory=list)
    thresholds: dict[str, float] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    exit_code: int = 0
    created_at: str = ""

    def to_dict(self) -> dict:
        """Serialise the report to a plain JSON-friendly dict."""
        return {
            "subject": self.subject,
            "judge": self.judge,
            "status": self.status,
            "exit_code": self.exit_code,
            "created_at": self.created_at,
            "n_cases": len(self.cases),
            "metrics": {k: _round_opt(v) for k, v in self.metrics.items()},
            "thresholds": dict(self.thresholds),
            "cases": [c.to_dict() for c in self.cases],
            "metadata": dict(self.metadata),
        }


def _round_opt(value: float | None, ndigits: int = 4) -> float | None:
    """Round a float (or pass through ``None``) for serialisation."""
    return round(value, ndigits) if value is not None else None