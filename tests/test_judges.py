"""Unit tests for the judges (evalforge.judges).

The Ollama judge is exercised with an injected fake ``http_post`` so the suite
stays fully hermetic — no network, no Ollama, no Docker.
"""

import pytest
import requests

from evalforge.judges import (
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_URL,
    HeuristicJudge,
    OllamaJudge,
    PerCaseJudge,
    _extract_json_object,
    _normalize,
    score_deterministic,
)
from evalforge.models import EvalCase, SubjectAnswer


def _case(**overrides) -> EvalCase:
    values = dict(
        id="c1",
        topic="reentrancy",
        question="What is a reentrancy attack?",
        expected_keywords=["reentrancy", "guard"],
        refuse=False,
        doc_ids=["aave-v3"],
    )
    values.update(overrides)
    return EvalCase(**values)


class TestScoreDeterministic:
    def test_good_answerable_answer(self):
        case = _case()
        answer = SubjectAnswer(
            answer="reentrancy guard fix",
            sources=["aave-v3 (page 41)"],
            retrieved_doc_ids=["aave-v3", "other"],
        )
        out = score_deterministic(case, answer)
        assert out.case_id == "c1"
        assert out.faith == pytest.approx(2 / 3)  # "fix" is outside the reference context
        assert out.rel == 1.0  # both keywords present
        assert out.cit == 1.0  # cited aave-v3 = expected
        assert out.p_at_k == 0.5  # 1 of 2 retrieved relevant
        assert out.rec == 1.0  # the only relevant doc was retrieved
        assert out.refusal_correct is True
        assert out.hallu is False  # answered with evidence
        assert out.reasoning == "lexical (token overlap)"
        assert out.error == ""

    def test_partial_answerable_answer(self):
        case = _case()
        answer = SubjectAnswer(answer="reentrancy only", sources=[])
        out = score_deterministic(case, answer)
        assert out.rel == 0.5  # 1 of 2 keywords
        assert out.cit == 0.0  # answered but cited nothing while docs expected
        assert out.refusal_correct is True

    def test_trap_refused_correctly(self):
        case = _case(expected_keywords=[], doc_ids=[], refuse=True)
        answer = SubjectAnswer(answer="I don't know", refused=True)
        out = score_deterministic(case, answer)
        assert out.faith == 1.0
        assert out.rel == 0.0
        assert out.cit is None
        assert out.refusal_correct is True
        assert out.hallu is False

    def test_trap_answered_is_fabrication(self):
        case = _case(expected_keywords=[], doc_ids=[], refuse=True)
        answer = SubjectAnswer(answer="the answer is 42")
        out = score_deterministic(case, answer)
        assert out.faith is None  # no reference context for a trap
        assert out.rel == 0.0
        assert out.refusal_correct is False
        assert out.hallu is True

    def test_grounded_refusal_answered_with_clarification(self):
        case = _case(expected_keywords=[], doc_ids=[], refuse=True)
        answer = SubjectAnswer(answer="please provide the ticker symbol", refused=True)
        out = score_deterministic(case, answer)
        assert out.refusal_correct is True
        assert out.hallu is False

    def test_empty_answer_is_refusal(self):
        case = _case()
        answer = SubjectAnswer(answer="")
        out = score_deterministic(case, answer)
        assert out.faith == 1.0  # vacuous
        assert out.refusal_correct is False  # answerable case refused

    def test_lexical_refusal_independent_of_adapter_mark(self):
        case = _case()
        answer = SubjectAnswer(answer="i do not know the answer", refused=False)
        out = score_deterministic(case, answer)
        assert out.refusal_correct is False
        assert out.hallu is False


class TestHeuristicJudge:
    def test_name(self):
        assert HeuristicJudge().name == "heuristic"

    def test_score_delegates(self):
        judge = HeuristicJudge()
        case = _case()
        out = judge.score(case, SubjectAnswer(answer="reentrancy guard fix"))
        assert isinstance(out, PerCaseJudge)
        assert out.rel == 1.0


class TestExtractJsonObject:
    def test_direct_json(self):
        assert _extract_json_object('{"faithfulness": 4, "relevance": 3}') == {
            "faithfulness": 4,
            "relevance": 3,
        }

    def test_fenced_block(self):
        text = 'Here you go:\n```json\n{"faithfulness": 4, "relevance": 3}\n```'
        assert _extract_json_object(text)["relevance"] == 3

    def test_text_wrapped_braces(self):
        text = 'Sure! The score is {"faithfulness": 4, "relevance": 3} hope it helps'
        assert _extract_json_object(text)["faithfulness"] == 4

    def test_trailing_commas_repaired(self):
        assert _extract_json_object('{"faithfulness": 4, "relevance": 3,}')["relevance"] == 3

    def test_double_brace_peel(self):
        text = '{{"faithfulness": 4, "relevance": 3}}'
        assert _extract_json_object(text)["faithfulness"] == 4

    def test_non_dict_json_is_rejected(self):
        assert _extract_json_object("[1, 2, 3]") is None

    def test_garbage_is_none(self):
        assert _extract_json_object("") is None
        assert _extract_json_object("no json here") is None


class TestNormalize:
    def test_mid_scale(self):
        assert _normalize(4) == pytest.approx(0.8)

    def test_clamps_above(self):
        assert _normalize(6) == 1.0

    def test_clamps_below(self):
        assert _normalize(-1) == 0.0

    def test_non_numeric_is_zero(self):
        assert _normalize("high") == 0.0


class _FakeResponse:
    def __init__(self, content="", *, raise_on_status=None, payload=None):
        self._content = content
        self._raise = raise_on_status
        self._payload = payload

    def raise_for_status(self):
        if self._raise is not None:
            raise self._raise

    def json(self):
        if self._payload is not None:
            return self._payload
        return {"message": {"content": self._content}}


def _post_factory(*items):
    """Build a fake ``http_post`` that returns/raises items in order."""
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        item = items[len(calls) - 1] if len(calls) <= len(items) else items[-1]
        if isinstance(item, Exception):
            raise item
        return item

    return post, calls


class TestOllamaJudge:
    def test_defaults(self):
        judge = OllamaJudge(http_post=lambda *a, **k: _FakeResponse())
        assert judge.base_url == DEFAULT_OLLAMA_URL
        assert judge.model == DEFAULT_OLLAMA_MODEL
        assert judge.max_attempts == 2

    def test_explicit_config(self):
        judge = OllamaJudge(base_url="http://ollama:11434/", model="qwen3:8b", http_post=lambda *a, **k: _FakeResponse())
        assert judge.base_url == "http://ollama:11434"  # trailing slash stripped
        assert judge.model == "qwen3:8b"

    def test_happy_path_json(self):
        post, calls = _post_factory(_FakeResponse('{"faithfulness": 4, "relevance": 3, "reasoning": "ok"}'))
        judge = OllamaJudge(http_post=post)
        out = judge.score(_case(), SubjectAnswer(answer="reentrancy guard fix"))
        assert out.faith == pytest.approx(0.8)
        assert out.rel == pytest.approx(0.6)
        assert out.reasoning == "ok"
        assert out.error == ""
        assert len(calls) == 1
        url, kwargs = calls[0]
        assert url.endswith("/api/chat")
        assert kwargs["json"]["model"] == DEFAULT_OLLAMA_MODEL
        assert kwargs["json"]["options"]["temperature"] == 0.0

    def test_fenced_response(self):
        post, _ = _post_factory(_FakeResponse('```json\n{"faithfulness": 5, "relevance": 4}\n```'))
        judge = OllamaJudge(http_post=post)
        out = judge.score(_case(), SubjectAnswer(answer="x"))
        assert out.faith == 1.0
        assert out.rel == pytest.approx(0.8)

    def test_malformed_retries_and_succeeds(self):
        post, calls = _post_factory(
            _FakeResponse("not json at all"),
            _FakeResponse('{"faithfulness": 4, "relevance": 3}'),
        )
        judge = OllamaJudge(http_post=post)
        out = judge.score(_case(), SubjectAnswer(answer="x"))
        assert out.faith == pytest.approx(0.8)
        assert len(calls) == 2  # one retry with a fresh request

    def test_http_exception_falls_back_to_deterministic(self):
        post, calls = _post_factory(
            requests.ConnectionError("boom"),
            requests.ConnectionError("boom again"),
        )
        judge = OllamaJudge(http_post=post)
        case = _case()
        out = judge.score(case, SubjectAnswer(answer="reentrancy guard fix"))
        assert out.error
        assert "ConnectionError" in out.error
        assert out.faith == pytest.approx(2 / 3)  # deterministic fallback kept
        assert out.rel == 1.0
        assert out.cit == 0.0  # answered with no sources while docs expected
        assert len(calls) == 2

    def test_raise_for_status_failure(self):
        post, _ = _post_factory(
            _FakeResponse(raise_on_status=RuntimeError("500")),
            _FakeResponse(raise_on_status=RuntimeError("500")),
        )
        judge = OllamaJudge(http_post=post)
        out = judge.score(_case(), SubjectAnswer(answer="x"))
        assert out.error
        assert "500" in out.error

    def test_missing_relevance_key_is_error(self):
        post, _ = _post_factory(_FakeResponse('{"faithfulness": 4}'))
        judge = OllamaJudge(http_post=post)
        out = judge.score(_case(), SubjectAnswer(answer="x"))
        assert out.error
        assert "missing faithfulness/relevance" in out.error

    def test_non_dict_output_is_error(self):
        post, _ = _post_factory(_FakeResponse("[1, 2, 3]"))
        judge = OllamaJudge(http_post=post)
        out = judge.score(_case(), SubjectAnswer(answer="x"))
        assert out.error
        assert "did not parse" in out.error

    def test_success_after_timeout_retry(self):
        post, calls = _post_factory(
            TimeoutError("read timeout"),
            _FakeResponse('{"faithfulness": 2, "relevance": 2}'),
        )
        judge = OllamaJudge(http_post=post)
        out = judge.score(_case(), SubjectAnswer(answer="x"))
        assert out.faith == pytest.approx(0.4)
        assert out.error == ""
        assert len(calls) == 2

    def test_never_reuses_request_object(self):
        seen: list[dict] = []

        def post(url, **kwargs):
            # kwargs["json"] must be a *fresh* dict per attempt
            seen.append(kwargs["json"])
            raise requests.ConnectionError("retry")

        judge = OllamaJudge(http_post=post, max_attempts=3)
        judge.score(_case(), SubjectAnswer(answer="x"))
        assert len(seen) == 3
        assert seen[0] is not seen[1]

    def test_ollama_judge_name(self):
        assert OllamaJudge(http_post=lambda *a, **k: _FakeResponse()).name == "ollama"

    def test_llm_score_only_overrides_faith_rel(self):
        post, _ = _post_factory(_FakeResponse('{"faithfulness": 5, "relevance": 5}'))
        judge = OllamaJudge(http_post=post)
        case = _case()
        answer = SubjectAnswer(
            answer="reentrancy guard fix",
            sources=["aave-v3 (page 41)"],
            retrieved_doc_ids=["aave-v3", "other"],
        )
        out = judge.score(case, answer)
        # Methodology metrics stay deterministic; only faith/rel come from the LLM.
        assert out.cit == 1.0
        assert out.p_at_k == 0.5
        assert out.rec == 1.0
        assert out.faith == 1.0
        assert out.rel == 1.0


class TestJudgeProtocol:
    """Both judges satisfy the Judge protocol (name + score)."""

    @pytest.mark.parametrize("judge", [HeuristicJudge(), OllamaJudge(http_post=lambda *a, **k: _FakeResponse())])
    def test_protocol_shape(self, judge):
        assert isinstance(judge.name, str)
        out = judge.score(_case(), SubjectAnswer(answer="reentrancy guard fix"))
        assert isinstance(out, PerCaseJudge)