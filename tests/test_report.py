"""Unit tests for report rendering and persistence (evalforge.report)."""

import json

import pytest

from evalforge.dataset import EvalDataset
from evalforge.models import EvalCase, SubjectAnswer
from evalforge.report import render_json, render_markdown, write_report
from evalforge.runner import run_eval


class _FakeSubject:
    name = "fake"

    def __init__(self, answers):
        self._answers = list(answers)

    def ask(self, question):
        return self._answers.pop(0)


def _case(cid: str, *, keywords, refuse=False, doc_ids=(), sources=None) -> EvalCase:
    return EvalCase(
        id=cid,
        topic="reentrancy",
        question=f"question {cid}",
        expected_keywords=list(keywords),
        refuse=refuse,
        doc_ids=list(doc_ids),
    )


def _pass_report():
    """A healthy run where every measurable metric clears its guard."""
    cases = [
        _case(
            "a1",
            keywords=["reentrancy", "attack", "mitigation"],
            doc_ids=["aave-v3", "balancer-managedpool"],
        )
    ]
    answer = SubjectAnswer(
        answer="reentrancy attack mitigation",
        sources=["aave-v3 (page 41)"],
        retrieved_doc_ids=["aave-v3", "balancer-managedpool"],
    )
    return run_eval(_FakeSubject([answer]), cases)


def _fail_report():
    """A run where answer_rate breaches its floor."""
    case = _case("a1", keywords=["reentrancy"])
    return run_eval(
        _FakeSubject([SubjectAnswer(answer="I don't know", refused=True)]),
        [case],
    )


class TestRenderMarkdown:
    def test_header_and_structure(self):
        md = render_markdown(_pass_report())
        assert "# Evalforge Evaluation Report" in md
        assert "**Subject**: `fake`" in md
        assert "## Verdict: **PASS**" in md
        assert "## Metrics" in md
        assert "## Per-case" in md
        assert "| faithfulness |" in md
        assert "| hallucination_rate |" in md

    def test_regression_guard_renders_for_measurable_metrics(self):
        md = render_markdown(_pass_report())
        assert "## Regression guard" in md
        assert "Thresholds applied to the measurable metrics:" in md
        assert "OK" in md
        # The bug-fix contract: metric names map to min_/max_ threshold keys.
        assert "| answer_rate | 1.0000 | 0.7 | OK |" in md

    def test_breach_marker_on_fail(self):
        md = render_markdown(_fail_report())
        assert "## Verdict: **FAIL**" in md
        assert "BREACH" in md

    def test_none_metrics_show_n_as_a(self):
        md = render_markdown(_pass_report())
        assert "| citation_accuracy |" in md  # present...
        # ...but for a run WITHOUT doc_ids the value is n/a
        case = _case("b1", keywords=["reentrancy"])
        report = run_eval(
            _FakeSubject([SubjectAnswer(answer="reentrancy")]),
            [case],
        )
        md_na = render_markdown(report)
        assert "| citation_accuracy | n/a |" in md_na
        assert "| context_precision | n/a |" in md_na

    def test_per_case_table_escapes_pipes(self):
        case = _case("a1", keywords=["reentrancy"])
        report = run_eval(
            _FakeSubject([SubjectAnswer(answer="reentrancy", refused=True)]),
            [case],
        )
        # force an error string with a pipe through a bad subject
        class _Pipe:
            name = "pipe"

            def ask(self, question):
                raise RuntimeError("boom | pipe")

        report = run_eval(_Pipe(), [case])
        md = render_markdown(report)
        assert "boom / pipe" in md  # pipe replaced, table still valid

    def test_metadata_line(self):
        md = render_markdown(_pass_report())
        assert "Run metadata:" in md


class TestRenderJson:
    def test_parses_and_has_shape(self):
        payload = json.loads(render_json(_pass_report()))
        assert payload["subject"] == "fake"
        assert payload["status"] == "PASS"
        assert payload["exit_code"] == 0
        assert payload["n_cases"] == 1
        assert payload["metrics"]["answer_rate"] == 1.0
        assert payload["metrics"]["citation_accuracy"] == 1.0
        assert payload["metrics"]["context_precision"] == 1.0
        assert payload["cases"][0]["case_id"] == "a1"

    def test_none_metrics_pass_through_as_null(self):
        case = _case("b1", keywords=["reentrancy"])
        report = run_eval(
            _FakeSubject([SubjectAnswer(answer="reentrancy")]),
            [case],
        )
        payload = json.loads(render_json(report))
        assert payload["metrics"]["citation_accuracy"] is None
        assert payload["metrics"]["hallucination_rate"] is None

    def test_unicode_safe(self):
        dataset = EvalDataset.from_cases(
            [{"id": "u1", "topic": "t", "question": "¿qué?", "expected_keywords": ["reentrancy"]}]
        )
        report = run_eval(
            _FakeSubject([SubjectAnswer(answer="reentrancy")]),
            list(dataset),
        )
        text = render_json(report)
        assert "¿qué?" in text


class TestWriteReport:
    def test_writes_md_and_json(self, tmp_path):
        report = _pass_report()
        md_path, json_path = write_report(report, tmp_path)
        assert md_path.exists()
        assert json_path.exists()
        assert md_path.suffix == ".md"
        assert json_path.suffix == ".json"
        assert "fake" in md_path.name
        assert "heuristic" in md_path.name

    def test_written_json_is_parseable(self, tmp_path):
        report = _pass_report()
        _, json_path = write_report(report, tmp_path)
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert payload["status"] == "PASS"

    def test_nested_dir_created(self, tmp_path):
        report = _pass_report()
        write_report(report, tmp_path / "deep" / "nested")
        assert (tmp_path / "deep" / "nested").is_dir()