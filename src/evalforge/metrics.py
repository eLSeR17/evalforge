"""Deterministic, lexical evaluation metrics (generalised).

This module is a *generalised port* of the metric semantics from the sibling
project ``smart-contract-rag`` (``smart_contract_rag.evals.metrics``). The
original formulas are preserved wherever they apply, but every dependency on
the RAG domain (retrieved chunk objects, document ids, grounding checks) has
been replaced with plain strings and lists so that any LLM pipeline/agent can
be scored.

The eight metrics produced by an eval run:

1. ``faithfulness`` — is the answer *contained* in the golden reference
   context (expected keywords + declared document ids + cited sources)?
   A conservative lexical proxy for "did the answer stay within the expected
   evidence", mirroring the original answer-tokens-in-retrieved-chunks ratio.
2. ``answer_relevance`` — fraction of the golden ``expected_keywords`` present
   in the answer (identical formula to the reference project).
3. ``citation_accuracy`` — do the cited sources include an expected document?
4. ``context_precision`` (p@k) — of the retrieved docs, how many are relevant.
5. ``context_recall`` — of the relevant docs, how many were retrieved.
6. ``answer_rate`` — fraction of answerable questions actually answered.
7. ``correct_refusal_rate`` — fraction of refusal decisions that were correct.
8. ``hallucination_rate`` — fraction of fabrication signals (answered where a
   refusal was expected, or answered without exposing any grounding evidence).

Per-case metrics that *cannot* be measured are ``None`` (not ``0.0``) and are
excluded from the aggregate — this is the documented difference from the
reference project, which silently passed unmeasurable metrics as vacuous 1.0.

All tokenisation is deliberately simple (lowercased word/identifier tokens)
so the metrics are dependency-free, deterministic, and fully reproducible.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

# Stopwords are ignored for faithfulness: they are shared by every sentence
# and would inflate the containment ratio without adding evidence signal.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
        "is", "are", "was", "were", "be", "been", "it", "its", "this", "that",
        "as", "at", "by", "from", "we", "you", "they", "i", "not", "but",
        "if", "so", "no", "yes", "do", "does", "did", "can", "could", "should",
        "will", "would", "may", "might", "have", "has", "had", "there", "which",
        "than", "about", "against", "between", "over", "into", "during",
    }
)

# Case-insensitive refusal phrases. Mirrors the reference project's signals,
# plus the explicit markers emitted by the smart-contract-rag CLI.
_REFUSAL_SIGNALS: tuple[str, ...] = (
    "i don't know",
    "i do not know",
    "i dont know",
    "dont know",
    "not enough",
    "no relevant sources",
    "response was refused pending grounding verification",
    "i'm sorry, but i cannot process this request",
    "i am sorry, but i cannot process this request",
)

_TOKEN_RE = re.compile(r"[a-z0-9_.]+")
_WORD_RE = re.compile(r"[a-z0-9_]+")

# Leading token of a source label. Citation accuracy extracts the document id
# from a label such as "aave-v3 (page 41)" -> "aave-v3", or from a bare id.
_SOURCE_ID_RE = re.compile(r"^\s*([A-Za-z0-9._-]+)")


def tokens(text: str) -> set[str]:
    """Lowercased substantive tokens (stopwords removed, numbers kept)."""
    out: set[str] = set()
    for match in _TOKEN_RE.finditer(text.lower()):
        token = match.group(0).strip(".")
        if not token:
            continue
        is_number = token.replace(".", "", 1).isdigit()
        if is_number or (len(token) >= 3 and token not in _STOPWORDS):
            out.add(token)
    return out


def word_tokens(text: str) -> set[str]:
    """Every lowercased word token, including stopwords (for keyword overlap)."""
    return set(_WORD_RE.findall(text.lower()))


def is_refusal(answer: str) -> bool:
    """Return True if the answer is a refusal (declines or is empty)."""
    lowered = answer.lower().strip()
    if not lowered:
        return True
    return any(signal in lowered for signal in _REFUSAL_SIGNALS)


def extract_doc_id(source_label: str) -> str:
    """Extract the document id from a source label.

    Accepts both bare ids (``"aave-v3"``) and labelled citations
    (``"aave-v3 (page 41)"``). The leading token is taken as the doc id — the
    same token shape the smart-contract-rag CLI emits for its source lines.
    """
    match = _SOURCE_ID_RE.match(source_label)
    return match.group(1) if match else source_label.strip()


# ---------------------------------------------------------------------------
# Per-case metrics
# ---------------------------------------------------------------------------
def faithfulness(answer: str, *, ref_context: str | None) -> float | None:
    """Lexical containment of the answer within the golden reference context.

    The *reference context* is the union of the case's expected keywords, its
    declared ``doc_ids``, and the sources the subject exposed. The metric is
    the fraction of the answer's substantive tokens that appear in that
    context — "did the answer stay inside the expected vocabulary/evidence".

    An empty or refusal answer is vacuously faithful (``1.0``: nothing to
    substantiate). Returns ``None`` when there is no reference context to
    measure against (e.g. an answered trap with no keywords).
    """
    if not answer.strip() or is_refusal(answer):
        return 1.0
    if not ref_context or not ref_context.strip():
        return None
    answer_terms = tokens(answer)
    if not answer_terms:
        return 1.0
    ref_terms = tokens(ref_context)
    if not ref_terms:
        return None
    contained = sum(1 for term in answer_terms if term in ref_terms)
    return contained / len(answer_terms)


def answer_relevance(
    answer: str,
    *,
    expected_keywords: Iterable[str] | None = None,
    ground_truth: str | None = None,
) -> float:
    """Lexical overlap of the answer with the golden expectation.

    If ``expected_keywords`` are provided they are the primary signal (fraction
    that appear in the answer). Otherwise the answer is compared against the
    ``ground_truth`` reference text by word overlap (kept for compatibility
    with the reference project; evalforge golden cases carry keywords).

    A refusal addresses the question by declining and is treated as neutral
    (``0.0``); the runner tracks refusal correctness separately.
    """
    if is_refusal(answer):
        return 0.0
    answer_tokens = word_tokens(answer)

    keywords = list(expected_keywords) if expected_keywords is not None else []
    if keywords:
        keyword_tokens: set[str] = set()
        for keyword in keywords:
            keyword_tokens |= word_tokens(keyword)
        if not keyword_tokens:
            return 0.0
        matched = keyword_tokens & answer_tokens
        return len(matched) / len(keyword_tokens)

    if ground_truth:
        truth_tokens = word_tokens(ground_truth)
        if not truth_tokens:
            return 0.0
        return len(truth_tokens & answer_tokens) / len(truth_tokens)

    return 0.0


def citation_accuracy(
    cited_sources: Sequence[str],
    expected_doc_ids: Sequence[str],
    *,
    refused: bool,
) -> float | None:
    """Whether the answer cites a document that actually backs it.

    Returns ``None`` when the case declares no expected documents or the
    answer was a refusal. Otherwise ``1.0`` if at least one cited source's
    document id matches an expected id, ``0.0`` if it cites something else
    (or cites nothing while an answer was expected).
    """
    if not expected_doc_ids:
        return None
    if refused:
        return None
    expected = set(expected_doc_ids)
    cited = {extract_doc_id(label) for label in cited_sources}
    return 1.0 if (cited & expected) else 0.0


def context_precision(
    retrieved_doc_ids: Sequence[str], expected_doc_ids: Sequence[str]
) -> float | None:
    """Precision@k: share of the retrieved documents that are relevant.

    ``k`` is the number of retrieved documents (the subject's full retrieval
    list). Returns ``None`` when the subject exposed no retrieval or the case
    declares no expected documents — the metric cannot be measured.
    """
    if not expected_doc_ids or not retrieved_doc_ids:
        return None
    retrieved = set(retrieved_doc_ids)
    hits = len(retrieved & set(expected_doc_ids))
    return hits / len(retrieved)


def context_recall(
    retrieved_doc_ids: Sequence[str], expected_doc_ids: Sequence[str]
) -> float | None:
    """Recall: fraction of the expected documents that were retrieved.

    Returns ``None`` when the subject exposed no retrieval or the case
    declares no expected documents.
    """
    if not expected_doc_ids or not retrieved_doc_ids:
        return None
    retrieved = set(retrieved_doc_ids)
    hits = len(retrieved & set(expected_doc_ids))
    return hits / len(set(expected_doc_ids))


def correct_refusal(*, refused: bool, expected_refusal: bool) -> bool:
    """Did the subject refuse *appropriately* for this case?

    ``expected_refusal`` is the case's ``refuse`` flag. Answerable cases are
    correct when they answer; refusal cases (grounded refusals and traps) are
    correct when they refuse.
    """
    return refused == expected_refusal


def hallucination(*, refused: bool, expected_refusal: bool, has_evidence: bool) -> bool | None:
    """Is the response a fabrication signal?

    - A refusal invents nothing -> ``False``.
    - Answering a case that expected a refusal is fabrication by construction
      (traps have no backing data) -> ``True``.
    - Answering with *no* exposed evidence cannot be assessed lexically
      -> ``None`` (excluded from the aggregate).
    - Otherwise -> ``False`` (the answer at least exposes grounding evidence;
      deeper semantic hallucination checking is the LLM judge's job).
    """
    if refused:
        return False
    if expected_refusal:
        return True
    if not has_evidence:
        return None
    return False


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
METRIC_NAMES: tuple[str, ...] = (
    "faithfulness",
    "answer_relevance",
    "citation_accuracy",
    "context_precision",
    "context_recall",
    "answer_rate",
    "correct_refusal_rate",
    "hallucination_rate",
)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate(case_metrics: Sequence[dict]) -> dict[str, float | None]:
    """Aggregate per-case metric dicts into the eight report metrics.

    Each input dict has keys: answered, refused, faith, rel, cit, p_at_k, rec,
    refusal_correct, hallu, and the case's ``refuse``/``expected_refusal``
    flag. Aggregates are means over the cases where the per-case value is not
    ``None``; a metric with no measurable case is ``None`` (excluded from the
    verdict and shown as ``n/a`` in the report).
    """
    answered_cases = [c for c in case_metrics if c["answered"]]

    faith_values = [c["faith"] for c in answered_cases if c["faith"] is not None]
    rel_values = [c["rel"] for c in answered_cases if c["rel"] is not None]
    cit_values = [c["cit"] for c in case_metrics if c["cit"] is not None]
    prec_values = [c["p_at_k"] for c in case_metrics if c["p_at_k"] is not None]
    rec_values = [c["rec"] for c in case_metrics if c["rec"] is not None]

    # Answer rate: fraction of answerable questions actually answered.
    # Refusal cases are excluded (they are expected to refuse).
    answerable = [c for c in case_metrics if not c["expected_refusal"]]
    answer_rate_values = [1.0 if c["answered"] else 0.0 for c in answerable]

    refusal_values = [1.0 if c["refusal_correct"] else 0.0 for c in case_metrics]
    hallu_values = [1.0 if c["hallu"] else 0.0 for c in case_metrics if c["hallu"] is not None]

    return {
        "faithfulness": _opt_mean(faith_values),
        "answer_relevance": _opt_mean(rel_values),
        "citation_accuracy": _opt_mean(cit_values),
        "context_precision": _opt_mean(prec_values),
        "context_recall": _opt_mean(rec_values),
        "answer_rate": _opt_mean(answer_rate_values),
        "correct_refusal_rate": _opt_mean(refusal_values),
        "hallucination_rate": _opt_mean(hallu_values),
    }


def _opt_mean(values: Sequence[float]) -> float | None:
    """Mean of the values, or ``None`` when there is nothing to average."""
    if not values:
        return None
    return _mean(values)