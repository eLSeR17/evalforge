"""Hermetic tests for the in-network re-scoring path (evalforge.rejudge).

``rejudge`` re-scores the captured responses of an existing artifact with a
(different) judge without re-running the subject. These tests use a stub
judge and in-memory/tmp_path artifacts — no network, no Docker, no Ollama.

The honesty contract under test:
- cases without ``subject_answer`` are kept verbatim (never invented);
- cases that errored in the original run are kept verbatim;
- a failing judge propagates the reason into ``judge_reason`` (fallback
  scores come from the judge itself — OllamaJudge returns the deterministic
  ones — rejudge never fabricates);
- the case count is always preserved and the original artifact untouched;
- the CLI writes a NEW ``..._ollama-in-network_<ts>.{md,json}`` artifact.
"""

import json

import pytest

from evalforge import cli
from evalforge.judges import HeuristicJudge, PerCaseJudge, score_deterministic
from evalforge.models import EvalCase, SubjectAnswer
from evalforge.rejudge import RejudgeOutcome, rejudge_artifact
from evalforge.report import write_report
from evalforge.runner import run_eval
from evalforge.subjects import SubjectError


class _StubSubject:
    """Answers pop from a list in case order; deterministic for tests."""

    name = "stub"

    def __init__(self, answers):
        self._answers = list(answers)

    def ask(self, question):
        if not self._answers:
            raise SubjectError("no more answers prepared")
        return self._answers.pop(0)


def _case(cid: str, *, keywords, refuse=False, doc_ids=()) -> EvalCase:
    return EvalCase(
        id=cid,
        topic="reentrancy",
        question=f"question {cid}",
        expected_keywords=list(keywords),
        refuse=refuse,
        doc_ids=list(doc_ids),
    )


class _StubJudge:
    """Judge stub: fixed semantic scores, optional failure, optional warm-up."""

    name = "stub"

    def __init__(self, *, faith=1.0, rel=1.0, reasoning="semantic stub", error=""):
        self.model = "qwen2.5-coder:7b"
        self.keep_alive = "5m"
        self._faith = faith
        self._rel = rel
        self._reasoning = reasoning
        self._error = error

    def score(self, case: EvalCase, answer: SubjectAnswer) -> PerCaseJudge:
        # A real judge returns deterministic fallback values on failure; the
        # stub mimics that so rejudge's propagation logic is what is tested.
        base = score_deterministic(case, answer)
        if self._error:
            return PerCaseJudge(
                case_id=case.id,
                faith=base.faith,
                rel=base.rel,
                cit=base.cit,
                p_at_k=base.p_at_k,
                rec=base.rec,
                refusal_correct=base.refusal_correct,
                hallu=base.hallu,
                error=self._error,
            )
        return PerCaseJudge(
            case_id=case.id,
            faith=self._faith,
            rel=self._rel,
            cit=base.cit,
            p_at_k=base.p_at_k,
            rec=base.rec,
            refusal_correct=base.refusal_correct,
            hallu=base.hallu,
            reasoning=self._reasoning,
        )

    def warm_up(self):
        return True, ""


def _write_corpus(tmp_path, *, with_error: bool = False):
    """Write a golden file and return cases + prepared answers."""
    cases = [
        _case("r1", keywords=["reentrancy", "guard"], doc_ids=["aave-v3"]),
        _case("r2", keywords=["external", "call"]),
    ]
    answers = [
        SubjectAnswer(
            answer="reentrancy guard is the fix",
            sources=["aave-v3 (page 41)"],
            retrieved_doc_ids=["aave-v3"],
        ),
        SubjectAnswer(answer="external calls must be guarded"),
    ]
    if with_error:
        # First case answers, second case crashes at ask() time.
        class _Exploding(_StubSubject):
            name = "stub"

            def ask(self, question):
                if not self._answers:
                    raise SubjectError("no more answers prepared")
                answer = self._answers.pop(0)
                if question == "question r2":
                    raise SubjectError("crashed on r2")
                return answer

        subject: _StubSubject = _Exploding(answers)
        expected_error = True
    else:
        subject = _StubSubject(answers)
        expected_error = False

    golden_path = tmp_path / "golden.json"
    golden_path.write_text(
        json.dumps(
            [
                {
                    "id": c.id,
                    "topic": c.topic,
                    "question": c.question,
                    "expected_keywords": c.expected_keywords,
                    "refuse": c.refuse,
                    "doc_ids": c.doc_ids,
                }
                for c in cases
            ]
        ),
        encoding="utf-8",
    )
    return cases, answers, subject, str(golden_path), expected_error


def _make_artifact(tmp_path, *, with_error: bool = False):
    """Run a stub eval and persist the JSON artifact; return (path, dict)."""
    cases, answers, subject, _golden, _err = _write_corpus(
        tmp_path, with_error=with_error
    )
    report = run_eval(subject, cases, metadata={"mode": "test"})
    _md_path, json_path = write_report(report, tmp_path)
    artifact = json.loads(json_path.read_text(encoding="utf-8"))
    return json_path, artifact, cases, answers


class TestRejudgeModule:
    def test_rescores_all_cases_and_preserves_count(self, tmp_path):
        json_path, artifact, _, _ = _make_artifact(tmp_path)
        original_text = json_path.read_text(encoding="utf-8")

        judge = _StubJudge(faith=1.0, rel=1.0, reasoning="semantic stub")
        outcome = rejudge_artifact(artifact, judge=judge, artifact_path=str(json_path))

        assert isinstance(outcome, RejudgeOutcome)
        assert outcome.n_rejudged == 2
        assert outcome.n_skipped == 0
        assert len(outcome.report.cases) == 2  # case count intact
        assert outcome.report.subject == "stub"
        assert outcome.report.judge == "stub"
        assert all(c.judge_reason == "semantic stub" for c in outcome.report.cases)
        assert all(c.faith == 1.0 and c.rel == 1.0 for c in outcome.report.cases)
        # The captured snapshot survives the re-scoring.
        assert outcome.report.cases[0].subject_answer == "reentrancy guard is the fix"
        assert outcome.report.cases[0].golden_reference == "reentrancy guard aave-v3"
        # The original artifact is byte-for-byte untouched.
        assert json_path.read_text(encoding="utf-8") == original_text

    def test_errored_case_kept_verbatim(self, tmp_path):
        json_path, artifact, _cases, _answers = _make_artifact(
            tmp_path, with_error=True
        )
        # Sanity: the source artifact really has an errored case.
        assert artifact["cases"][1]["error"].startswith("subject error:")

        outcome = rejudge_artifact(artifact, judge=_StubJudge(), artifact_path=str(json_path))

        assert outcome.n_rejudged == 1
        assert outcome.n_skipped == 1
        assert "errored" in outcome.skipped_reasons[0]
        assert len(outcome.report.cases) == 2  # count intact
        skipped = outcome.report.cases[1]
        assert skipped.error.startswith("subject error:")  # kept verbatim
        assert skipped.judge_reason == artifact["cases"][1]["judge_reason"]
        # Re-scored case carries the semantic reason.
        assert outcome.report.cases[0].judge_reason == "semantic stub"
        assert outcome.report.cases[0].faith == 1.0

    def test_missing_subject_answer_is_skipped_not_invented(self, tmp_path):
        json_path, artifact, _cases, _answers = _make_artifact(tmp_path)
        # Simulate a pre-INC-004 artifact: the answer snapshot was not captured.
        for case in artifact["cases"]:
            case.pop("subject_answer", None)
            case.pop("subject_sources", None)
            case.pop("subject_refused", None)

        outcome = rejudge_artifact(artifact, judge=_StubJudge(), artifact_path=str(json_path))

        assert outcome.n_rejudged == 0
        assert outcome.n_skipped == 2
        assert all("no subject_answer" in r for r in outcome.skipped_reasons)
        assert len(outcome.report.cases) == 2
        # Every case keeps its original deterministic scores.
        assert outcome.report.cases[0].faith == artifact["cases"][0]["faithfulness"]
        assert outcome.report.cases[0].judge_reason == artifact["cases"][0]["judge_reason"]

    def test_judge_failure_propagates_reason(self, tmp_path):
        json_path, artifact, _cases, _answers = _make_artifact(tmp_path)
        judge = _StubJudge(error="simulated judge failure")

        outcome = rejudge_artifact(artifact, judge=judge, artifact_path=str(json_path))

        assert outcome.n_rejudged == 2
        for case in outcome.report.cases:
            assert case.judge_reason == "simulated judge failure"
            # Fallback semantics delegated to the judge: rejudge itself never
            # fabricates scores — it only propagates what the judge returned.

    def test_new_report_recomputes_metrics_and_verdict(self, tmp_path):
        json_path, artifact, _cases, _answers = _make_artifact(tmp_path)
        # Original heuristic run: r2's "external calls" misses the exact token
        # "call" -> relevance 0.5, aggregate 0.75 (still over the 0.6 floor).
        assert artifact["metrics"]["answer_relevance"] == 0.75

        harsh = _StubJudge(faith=0.2, rel=0.2, reasoning="harsh")
        outcome = rejudge_artifact(artifact, judge=harsh, artifact_path=str(json_path))

        assert outcome.report.metrics["faithfulness"] == 0.2
        assert outcome.report.metrics["answer_relevance"] == 0.2
        # Faithfulness floor 0.70 breached -> FAIL, exit code 1.
        assert outcome.report.status == "FAIL"
        assert outcome.report.exit_code == 1

    def test_thresholds_carried_over_from_artifact(self, tmp_path):
        json_path, artifact, _cases, _answers = _make_artifact(tmp_path)
        outcome = rejudge_artifact(artifact, judge=_StubJudge(), artifact_path=str(json_path))
        assert outcome.report.thresholds == artifact["thresholds"]
        assert "rejudge_of" in outcome.report.metadata
        assert outcome.report.metadata["original_judge"] == artifact["judge"]
        assert outcome.report.metadata["n_rejudged"] == 2

    def test_not_an_evalforge_report_raises(self, tmp_path):
        with pytest.raises(ValueError, match="no 'cases' list"):
            rejudge_artifact({"subject": "x"}, judge=_StubJudge())

    def test_heuristic_rejudge_keeps_scores_stable(self, tmp_path):
        # Re-scoring with the deterministic judge reproduces the same lexical
        # numbers — a sanity check that reconstruction does not distort.
        json_path, artifact, _cases, _answers = _make_artifact(tmp_path)
        outcome = rejudge_artifact(
            artifact, judge=HeuristicJudge(), artifact_path=str(json_path)
        )
        assert outcome.n_rejudged == 2
        for c, raw in zip(outcome.report.cases, artifact["cases"]):
            # The artifact stores 4-decimal-rounded values; recompute may hold
            # more precision, so round before comparing.
            assert round(c.faith, 4) == raw["faithfulness"]
            assert round(c.rel, 4) == raw["answer_relevance"]


class TestRejudgeCli:
    def test_writes_new_in_network_artifact(self, tmp_path, monkeypatch, capsys):
        json_path, _artifact, _cases, answers = _make_artifact(tmp_path)
        _golden = json_path.parent / "golden.json"

        class _OllamaStub(_StubJudge):
            """Mirrors the real OllamaJudge name so the artifact says ollama."""

            name = "ollama"

        stub = _OllamaStub()
        monkeypatch.setattr(cli, "OllamaJudge", lambda **kwargs: stub)

        exit_code = cli.main(
            [
                "rejudge",
                "--artifact", str(json_path),
                "--judge", "ollama",
                "--golden", str(_golden),
                "--model", "qwen2.5-coder:7b",
            ]
        )

        assert exit_code == 0  # all cases re-scored to perfect scores -> PASS
        out = capsys.readouterr()
        assert "[rejudge]" in out.err
        assert "re-scored 2 of 2 cases (0 skipped" in out.err
        # NEW artifact with the in-network label, never overwriting the source.
        new_jsons = sorted(
            tmp_path.glob("eval_report_stub_ollama-in-network_*.json")
        )
        assert len(new_jsons) == 1
        assert new_jsons[0] != json_path
        payload = json.loads(new_jsons[0].read_text(encoding="utf-8"))
        assert payload["judge"] == "ollama"
        assert payload["n_cases"] == 2
        assert payload["cases"][0]["subject_answer"] == answers[0].answer
        assert payload["cases"][0]["judge_reason"] == "semantic stub"
        # The source artifact is untouched.
        assert "ollama-in-network" not in json_path.name

    def test_missing_artifact_returns_fail(self, tmp_path, monkeypatch, capsys):
        exit_code = cli.main(
            ["rejudge", "--artifact", str(tmp_path / "nope.json"), "--judge", "ollama"]
        )
        assert exit_code == 1
        assert "artifact not found" in capsys.readouterr().err

    def test_legacy_artifact_without_answers_skips_all(self, tmp_path, monkeypatch, capsys):
        json_path, artifact, _cases, _answers = _make_artifact(tmp_path)
        for case in artifact["cases"]:
            case.pop("subject_answer", None)
            case.pop("subject_sources", None)
        json_path.write_text(json.dumps(artifact), encoding="utf-8")

        stub = _StubJudge()
        monkeypatch.setattr(cli, "OllamaJudge", lambda **kwargs: stub)

        exit_code = cli.main(
            [
                "rejudge",
                "--artifact", str(json_path),
                "--judge", "ollama",
                "--model", "qwen2.5-coder:7b",
                "--no-warm-up",
            ]
        )
        # Nothing re-scored; honest skip; verdict/exit code mirror the original
        # run (which FAILs on the lexical faithfulness floor).
        assert exit_code == 1
        err = capsys.readouterr().err
        assert "re-scored 0 of 2 cases (2 skipped" in err
        assert "no subject_answer" in err
        report = min(
            tmp_path.glob("eval_report_stub_ollama-in-network_*.json"),
            key=lambda p: p.name,
        )
        new_payload = json.loads(report.read_text(encoding="utf-8"))
        assert new_payload["n_cases"] == 2

    def test_no_warm_up_flag_skips_warm_up(self, tmp_path, monkeypatch):
        json_path, _artifact, _cases, _answers = _make_artifact(tmp_path)

        warmed = {"called": False}

        class _WarmStub(_StubJudge):
            def warm_up(self):
                warmed["called"] = True
                return True, ""

        monkeypatch.setattr(cli, "OllamaJudge", lambda **kwargs: _WarmStub())
        assert (
            cli.main(
                [
                    "rejudge",
                    "--artifact", str(json_path),
                    "--judge", "ollama",
                    "--no-warm-up",
                ]
            )
            == 0
        )
        assert warmed["called"] is False