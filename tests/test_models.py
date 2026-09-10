"""Unit tests for the core value objects (evalforge.models)."""

import pytest

from evalforge.models import (
    EvalCase,
    EvalReport,
    PerCaseResult,
    SubjectAnswer,
    _round_opt,
)


class TestEvalCase:
    def test_defaults(self):
        case = EvalCase(id="c1", topic="t", question="q")
        assert case.expected_keywords == []
        assert case.refuse is False
        assert case.doc_ids == []

    def test_full_construction(self):
        case = EvalCase(
            id="c1",
            topic="reentrancy",
            question="What is a reentrancy attack?",
            expected_keywords=["external", "call"],
            refuse=False,
            doc_ids=["aave-v3"],
        )
        assert case.id == "c1"
        assert case.expected_keywords == ["external", "call"]
        assert case.doc_ids == ["aave-v3"]

    def test_immutable(self):
        case = EvalCase(id="c1", topic="t", question="q")
        with pytest.raises(AttributeError):  # dataclasses.FrozenInstanceError
            case.question = "changed"  # type: ignore[misc]

    def test_str(self):
        case = EvalCase(id="c1", topic="t", question="q", refuse=True)
        assert "refuse" in str(case)


class TestSubjectAnswer:
    def test_defaults(self):
        answer = SubjectAnswer(answer="hello")
        assert answer.sources == []
        assert answer.retrieved_doc_ids == []
        assert answer.refused is False

    def test_str(self):
        answer = SubjectAnswer(answer="hello", refused=True)
        assert "refused" in str(answer)


class TestPerCaseResult:
    def test_to_dict(self):
        result = PerCaseResult(
            case_id="c1",
            topic="t",
            question="q",
            answered=True,
            refused=False,
            faith=0.5,
            rel=0.25,
            cit=None,
            p_at_k=1.0,
            rec=None,
            refusal_correct=True,
            hallu=False,
            error="",
            judge_reason="lexical",
        )
        data = result.to_dict()
        assert data["case_id"] == "c1"
        assert data["faithfulness"] == 0.5
        assert data["answer_relevance"] == 0.25
        assert data["citation_accuracy"] is None
        assert data["context_recall"] is None
        assert data["correct_refusal"] is True
        assert data["hallucination"] is False
        assert data["judge_reason"] == "lexical"

    def test_to_dict_hallucination_none(self):
        result = PerCaseResult(
            case_id="c1",
            topic="t",
            question="q",
            answered=True,
            refused=False,
            faith=1.0,
            rel=1.0,
            cit=None,
            p_at_k=None,
            rec=None,
            refusal_correct=True,
            hallu=None,
        )
        assert result.to_dict()["hallucination"] is None


class TestEvalReport:
    def test_to_dict(self):
        report = EvalReport(
            subject="fake",
            judge="heuristic",
            status="PASS",
            metrics={"answer_relevance": 1.0, "citation_accuracy": None},
            cases=[
                PerCaseResult(
                    case_id="c1",
                    topic="t",
                    question="q",
                    answered=True,
                    refused=False,
                    faith=1.0,
                    rel=1.0,
                    cit=None,
                    p_at_k=None,
                    rec=None,
                    refusal_correct=True,
                    hallu=None,
                )
            ],
            thresholds={"min_answer_rate": 0.7},
            metadata={"golden": "data/golden/fake.json"},
            exit_code=0,
            created_at="2026-09-08T12:00:00Z",
        )
        data = report.to_dict()
        assert data["subject"] == "fake"
        assert data["status"] == "PASS"
        assert data["exit_code"] == 0
        assert data["n_cases"] == 1
        assert data["metrics"]["answer_relevance"] == 1.0
        assert data["metrics"]["citation_accuracy"] is None
        assert data["thresholds"] == {"min_answer_rate": 0.7}
        assert data["metadata"] == {"golden": "data/golden/fake.json"}

    def test_defaults(self):
        report = EvalReport(subject="fake", judge="heuristic")
        assert report.status == "PASS"
        assert report.metrics == {}
        assert report.cases == []
        assert report.exit_code == 0


class TestRoundOpt:
    def test_rounds_floats(self):
        assert _round_opt(0.123456) == 0.1235

    def test_passes_none(self):
        assert _round_opt(None) is None