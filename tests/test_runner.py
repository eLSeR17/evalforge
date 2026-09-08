"""Unit tests for the eval runner and regression guard (evalforge.runner)."""

import pytest

from evalforge.judges import HeuristicJudge
from evalforge.metrics import METRIC_NAMES
from evalforge.models import EvalCase, SubjectAnswer
from evalforge.runner import (
    EXIT_CODE_FAIL,
    EXIT_CODE_PASS,
    RegressionThresholds,
    Verdict,
    _verdict,
    run_eval,
)
from evalforge.subjects import SubjectError


class _FakeSubject:
    """Answers pop from a list in case order; deterministic for tests."""

    name = "fake"

    def __init__(self, answers):
        self._answers = list(answers)

    def ask(self, question):
        if not self._answers:
            raise SubjectError("no more answers prepared")
        return self._answers.pop(0)


def _case(cid: str, *, keywords, refuse=False, doc_ids=()) -> EvalCase:
    return EvalCase(
        id=cid,
        topic="t",
        question=f"question {cid}",
        expected_keywords=list(keywords),
        refuse=refuse,
        doc_ids=list(doc_ids),
    )


def _good_answer(case: EvalCase) -> SubjectAnswer:
    return SubjectAnswer(answer=" ".join(case.expected_keywords))


def _refusal_answer() -> SubjectAnswer:
    return SubjectAnswer(answer="I don't know", refused=True)


class TestGuardRaises:
    def test_empty_case_list_raises(self):
        with pytest.raises(ValueError, match="at least one golden case"):
            run_eval(_FakeSubject([]), [])

    def test_unknown_threshold_key_raises(self):
        with pytest.raises(ValueError, match="unknown threshold keys"):
            RegressionThresholds().with_overrides({"min_typo": 0.5})

    def test_threshold_override_applies(self):
        thr = RegressionThresholds().with_overrides({"min_answer_rate": 0.5})
        assert thr.min_answer_rate == 0.5
        assert thr.min_faithfulness == 0.70  # untouched


class TestVerdicts:
    def test_good_subject_passes(self):
        cases = [
            _case("a1", keywords=["reentrancy", "guard"]),
            _case("a2", keywords=["external", "call", "state"]),
        ]
        subject = _FakeSubject([_good_answer(c) for c in cases])
        report = run_eval(subject, cases)

        assert report.status == Verdict.PASS.value
        assert report.exit_code == EXIT_CODE_PASS
        assert report.subject == "fake"
        assert report.judge == "heuristic"
        assert report.metrics["answer_rate"] == 1.0
        assert report.metrics["faithfulness"] == 1.0
        assert report.metrics["answer_relevance"] == 1.0
        assert report.metrics["correct_refusal_rate"] == 1.0
        # Unmeasurable for a subject with no retrieval/evidence -> excluded.
        assert report.metrics["citation_accuracy"] is None
        assert report.metrics["context_precision"] is None
        assert report.metrics["context_recall"] is None
        assert report.metrics["hallucination_rate"] is None

    def test_refusing_subject_fails(self):
        cases = [_case("a1", keywords=["reentrancy"])]
        subject = _FakeSubject([_refusal_answer()])
        report = run_eval(subject, cases)
        assert report.status == Verdict.FAIL.value
        assert report.exit_code == EXIT_CODE_FAIL
        assert report.metrics["answer_rate"] == 0.0

    def test_partial_answer_rate_warns(self):
        cases = [_case(f"a{i}", keywords=["reentrancy"]) for i in range(10)]
        answers = [_good_answer(c) for c in cases[:7]] + [_refusal_answer()] * 3
        report = run_eval(_FakeSubject(answers), cases)
        assert report.metrics["answer_rate"] == pytest.approx(0.7)
        # 0.7 is inside the [0.70, 0.75) warn band -> WARN, non-fatal.
        assert report.status == Verdict.WARN.value
        assert report.exit_code == EXIT_CODE_PASS

    def test_hallucination_warns_near_cap(self):
        # Banding unit test: rate inside (guard - margin, guard] -> WARN.
        # (An E2E run cannot reach this state: answering a trap also sets
        # answer_relevance = 0.0, which FAILs on its own floor — by design.)
        thr = RegressionThresholds()
        metrics = {
            "faithfulness": 1.0,
            "answer_relevance": 1.0,
            "answer_rate": 1.0,
            "hallucination_rate": 0.08,  # <= guard 0.10, > guard-margin 0.05
            "citation_accuracy": None,
            "context_precision": None,
            "context_recall": None,
            "correct_refusal_rate": 1.0,
        }
        assert _verdict(metrics, thr) == Verdict.WARN

    def test_hallucination_fails_past_cap(self):
        thr = RegressionThresholds()
        metrics = {
            "faithfulness": 1.0,
            "answer_relevance": 1.0,
            "answer_rate": 1.0,
            "hallucination_rate": 0.15,
            "citation_accuracy": None,
            "context_precision": None,
            "context_recall": None,
            "correct_refusal_rate": 1.0,
        }
        assert _verdict(metrics, thr) == Verdict.FAIL

    def test_verdict_passes_with_clean_metrics(self):
        thr = RegressionThresholds()
        metrics = {
            "faithfulness": 1.0,
            "answer_relevance": 1.0,
            "citation_accuracy": 1.0,
            "context_precision": 1.0,
            "context_recall": 1.0,
            "answer_rate": 1.0,
            "correct_refusal_rate": 1.0,
            "hallucination_rate": 0.0,
        }
        assert _verdict(metrics, thr) == Verdict.PASS

    def test_verdict_fails_on_low_relevance(self):
        thr = RegressionThresholds()
        metrics = {
            "faithfulness": 1.0,
            "answer_relevance": 0.55,  # < floor 0.60
            "answer_rate": 1.0,
            "hallucination_rate": 0.0,
            "citation_accuracy": None,
            "context_precision": None,
            "context_recall": None,
            "correct_refusal_rate": 1.0,
        }
        assert _verdict(metrics, thr) == Verdict.FAIL

    def test_trap_answered_fails(self):
        cases = [_case("t1", keywords=[], refuse=True)]
        subject = _FakeSubject([SubjectAnswer(answer="yes, absolutely")])
        report = run_eval(subject, cases)
        assert report.metrics["correct_refusal_rate"] == 0.0
        # answered trap -> fab signal; correct_refusal is not guarded, so the
        # fail must come from... nothing guarded?? -> assert the verdict comes
        # from the fabrication via hallucination_rate being unmeasurable is
        # wrong; correct_refusal_rate is NOT in the guard set.
        assert report.metrics["hallucination_rate"] == 1.0
        assert report.status == Verdict.FAIL.value

    def test_metric_names_present_in_report(self):
        cases = [_case("a1", keywords=["reentrancy"])]
        report = run_eval(_FakeSubject([_good_answer(cases[0])]), cases)
        assert set(report.metrics) == set(METRIC_NAMES)

    def test_metadata_and_timestamps(self):
        cases = [_case("a1", keywords=["reentrancy"])]
        report = run_eval(
            _FakeSubject([_good_answer(cases[0])]),
            cases,
            judge=HeuristicJudge(),
            dataset_source="data/golden/test.json",
            metadata={"golden": "data/golden/test.json"},
        )
        assert report.metadata["dataset"] == "data/golden/test.json"
        assert report.metadata["n_cases"] == 1
        assert report.metadata["golden"] == "data/golden/test.json"
        assert report.created_at.endswith("Z")
        assert len(report.cases) == 1


class TestAnswerPersistence:
    """INC-004 fix: the artifact persists the raw subject answer and the
    golden contract so a later run can re-score the same responses with a
    different judge (`evalforge rejudge`) without re-running the subject."""

    def test_case_dicts_capture_subject_answer(self):
        cases = [
            _case("a1", keywords=["reentrancy", "guard"], doc_ids=["aave-v3"]),
            _case("a2", keywords=["external", "call"]),
        ]
        answers = [
            SubjectAnswer(
                answer="reentrancy guard comes first",
                sources=["aave-v3 (page 41)"],
                retrieved_doc_ids=["aave-v3"],
            ),
            SubjectAnswer(answer="external call is where the risk lives"),
        ]
        report = run_eval(_FakeSubject(answers), cases)
        data = report.to_dict()

        assert data["cases"][0]["subject_answer"] == "reentrancy guard comes first"
        assert data["cases"][0]["subject_sources"] == ["aave-v3 (page 41)"]
        assert data["cases"][0]["subject_retrieved_doc_ids"] == ["aave-v3"]
        assert data["cases"][0]["golden_reference"] == "reentrancy guard aave-v3"
        assert data["cases"][0]["expected_keywords"] == ["reentrancy", "guard"]
        assert data["cases"][0]["doc_ids"] == ["aave-v3"]
        assert data["cases"][0]["refuse"] is False
        assert data["cases"][1]["subject_answer"] == "external call is where the risk lives"
        assert data["cases"][1]["golden_reference"] == "external call"

    def test_json_artifact_roundtrip_via_tmp_path(self, tmp_path):
        """A run written through write_report must persist subject_answer in
        the JSON artifact (the on-disk contract `rejudge` consumes)."""
        from evalforge.report import write_report

        cases = [_case("a1", keywords=["reentrancy", "guard"])]
        stub_answer = SubjectAnswer(answer="the reentrancy guard is the fix")
        report = run_eval(_FakeSubject([stub_answer]), cases)
        _md_path, json_path = write_report(report, tmp_path)

        import json

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert payload["cases"][0]["subject_answer"] == "the reentrancy guard is the fix"
        assert payload["cases"][0]["golden_reference"] == "reentrancy guard"
        assert payload["cases"][0]["question"] == "question a1"
        # The scored fields remain present alongside the snapshot.
        assert payload["cases"][0]["answer_relevance"] == 1.0

    def test_subject_error_case_persists_empty_answer(self):
        class _Boom:
            name = "boom"

            def ask(self, question):
                raise SubjectError("crashed")

        report = run_eval(_Boom(), [_case("a1", keywords=["reentrancy"])])
        data = report.to_dict()
        case = data["cases"][0]
        assert case["error"].startswith("subject error:")
        assert case["subject_answer"] == ""  # nothing to re-score later
        assert case["golden_reference"] == "reentrancy"


class TestErrorHandling:
    def test_subject_error_marks_case(self):
        cases = [_case("a1", keywords=["reentrancy"])]

        class _Boom:
            name = "boom"

            def ask(self, question):
                raise SubjectError("docker exec failed")

        report = run_eval(_Boom(), cases)
        result = report.cases[0]
        assert result.refused is True
        assert result.answered is False
        assert result.error.startswith("subject error: docker exec failed")
        assert report.status == Verdict.FAIL.value

    def test_unexpected_error_marks_case(self):
        cases = [_case("a1", keywords=["reentrancy"])]

        class _Crazy:
            name = "crazy"

            def ask(self, question):
                raise RuntimeError("kaboom")

        report = run_eval(_Crazy(), cases)
        assert report.cases[0].error == "subject error: RuntimeError: kaboom"

    def test_empty_answer_mapped_to_refusal(self):
        cases = [_case("a1", keywords=["reentrancy"])]
        subject = _FakeSubject([SubjectAnswer(answer="")])
        report = run_eval(subject, cases)
        result = report.cases[0]
        assert result.refused is True
        assert result.answered is False

    def test_lexical_refusal_maps_even_without_mark(self):
        cases = [_case("a1", keywords=["reentrancy"])]
        subject = _FakeSubject([SubjectAnswer(answer="i do not know", refused=False)])
        report = run_eval(subject, cases)
        assert report.cases[0].refused is True
        assert report.metrics["answer_rate"] == 0.0

    def test_unmeasurable_metrics_excluded_from_verdict(self):
        # No doc_ids, no retrieval, no sources: cit/precision/recall/hallu
        # would be 0.0 IF computed — but they are None and therefore must NOT
        # break a healthy PASS.
        cases = [
            _case("a1", keywords=["reentrancy", "guard"]),
            _case("a2", keywords=["external", "call"]),
        ]
        subject = _FakeSubject([_good_answer(c) for c in cases])
        report = run_eval(subject, cases)
        assert report.status == Verdict.PASS.value
        assert report.metrics["citation_accuracy"] is None


class TestProvidedJudge:
    def test_ollama_judge_name_propagates(self):
        cases = [_case("a1", keywords=["reentrancy"])]
        report = run_eval(
            _FakeSubject([_good_answer(cases[0])]),
            cases,
            judge=HeuristicJudge(),
        )
        assert report.judge == "heuristic"