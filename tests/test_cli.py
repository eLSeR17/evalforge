"""Unit tests for the command-line entry point (evalforge.cli).

Everything is hermetic: `build_subject` is monkeypatched, the Ollama judge is
replaced, and report artifacts go to a tmp_path (never data/reports/ in the
repo during tests).
"""

import json

import pytest

from evalforge import cli
from evalforge.judges import HeuristicJudge
from evalforge.models import SubjectAnswer
from evalforge.runner import RegressionThresholds
from evalforge.subjects import SubjectError


class _FakeSubject:
    """Answers whatever the test prepared, in question order."""

    def __init__(self, answers):
        self.name = "fake"
        self._answers = list(answers)

    def ask(self, question):
        if not self._answers:
            raise SubjectError("no answers prepared")
        return self._answers.pop(0)


def _write_golden(tmp_path, n_cases: int) -> str:
    """Write a golden dataset with ``n_cases`` answerable cases."""
    cases = [
        {
            "id": f"t{i:03d}",
            "topic": "reentrancy",
            "question": f"What is reentrancy case {i}?",
            "expected_keywords": ["reentrancy", "guard"],
            "refuse": False,
            "doc_ids": [],
        }
        for i in range(n_cases)
    ]
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(cases), encoding="utf-8")
    return str(path)


def _good_answer():
    return SubjectAnswer(answer="reentrancy guard")


def _refusal():
    return SubjectAnswer(answer="I don't know", refused=True)


def _make_args(tmp_path, n_cases=3, extra=()):
    return [
        "--subject", "alpha-agent",
        "--golden", _write_golden(tmp_path, n_cases),
        "--report-dir", str(tmp_path),
        *extra,
    ]


class TestThresholdParsing:
    def test_empty(self):
        assert cli._parse_threshold_overrides(None) == {}
        assert cli._parse_threshold_overrides("") == {}

    def test_single(self):
        assert cli._parse_threshold_overrides("min_answer_rate=0.8") == {
            "min_answer_rate": 0.8
        }

    def test_multiple_with_spaces(self):
        assert cli._parse_threshold_overrides("a=1.5, b=2, c=0.05") == {
            "a": 1.5,
            "b": 2.0,
            "c": 0.05,
        }

    def test_missing_equals_raises(self):
        with pytest.raises(ValueError, match="expected key=value"):
            cli._parse_threshold_overrides("min_answer_rate")

    def test_bad_value_raises(self):
        with pytest.raises(ValueError, match="invalid threshold value"):
            cli._parse_threshold_overrides("min_answer_rate=fast")

    def test_rounds(self):
        assert cli._parse_threshold_overrides("min_answer_rate=7") == {
            "min_answer_rate": 7.0
        }


class TestParser:
    def test_defaults_to_heuristic(self):
        parser = cli.build_parser()
        args = parser.parse_args(["--subject", "alpha-agent"])
        assert args.judge == "heuristic"
        assert args.report_dir == "data/reports"

    def test_ollama_choice(self):
        parser = cli.build_parser()
        args = parser.parse_args(["--subject", "alpha-agent", "--judge", "ollama"])
        assert args.judge == "ollama"

    def test_subject_is_required(self):
        parser = cli.build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])


class TestDefaultGolden:
    """CLI default golden resolution follows the snake_case convention."""

    def test_kebab_subject_resolves_to_snake_case_file(self):
        assert cli._default_golden("alpha-agent") == "data/golden/alpha_agent.json"
        assert cli._default_golden("smart-contract-rag") == "data/golden/smart_contract_rag.json"

    def test_plain_subject_unchanged(self):
        assert cli._default_golden("foo") == "data/golden/foo.json"


class TestCliRun:
    def test_pass_exit_zero(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(cli, "build_subject", lambda n: _FakeSubject([_good_answer()] * 3))
        assert cli.main(_make_args(tmp_path)) == 0
        out = capsys.readouterr().out
        assert "## Verdict: **PASS**" in out

    def test_fail_exit_one(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(cli, "build_subject", lambda n: _FakeSubject([_refusal()] * 3))
        assert cli.main(_make_args(tmp_path)) == 1
        assert "## Verdict: **FAIL**" in capsys.readouterr().out

    def test_warn_is_non_fatal(self, tmp_path, monkeypatch, capsys):
        # 7/10 answered -> answer_rate 0.7 -> WARN -> exit 0.
        answers = [_good_answer()] * 7 + [_refusal()] * 3
        monkeypatch.setattr(cli, "build_subject", lambda n: _FakeSubject(answers))
        assert cli.main(_make_args(tmp_path, n_cases=10)) == 0
        assert "## Verdict: **WARN**" in capsys.readouterr().out

    def test_threshold_flag_can_turn_fail_into_pass(self, tmp_path, monkeypatch):
        # 2 answered of 3 -> default floor 0.70 -> FAIL
        answers = [_good_answer(), _good_answer(), _refusal()]
        monkeypatch.setattr(cli, "build_subject", lambda n: _FakeSubject(answers))
        assert cli.main(_make_args(tmp_path, n_cases=3)) == 1
        # with a relaxed floor the same run passes
        monkeypatch.setattr(cli, "build_subject", lambda n: _FakeSubject(answers))
        assert (
            cli.main(
                _make_args(tmp_path, n_cases=3, extra=["--threshold", "min_answer_rate=0.5"])
            )
            == 0
        )

    def test_env_threshold_applies(self, tmp_path, monkeypatch):
        monkeypatch.setenv("EVALFORGE_THRESHOLD", "min_answer_rate=0.5")
        answers = [_good_answer(), _good_answer(), _refusal()]
        monkeypatch.setattr(cli, "build_subject", lambda n: _FakeSubject(answers))
        assert cli.main(_make_args(tmp_path, n_cases=3)) == 0

    def test_cli_threshold_wins_over_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("EVALFORGE_THRESHOLD", "min_answer_rate=0.1")
        answers = [_good_answer(), _good_answer(), _refusal()]
        monkeypatch.setattr(cli, "build_subject", lambda n: _FakeSubject(answers))
        # env floor 0.1 -> PASS; CLI floor 0.99 -> FAIL
        assert cli.main(_make_args(tmp_path, n_cases=3)) == 0
        monkeypatch.setattr(cli, "build_subject", lambda n: _FakeSubject(answers))
        assert (
            cli.main(
                _make_args(tmp_path, n_cases=3, extra=["--threshold", "min_answer_rate=0.99"])
            )
            == 1
        )

    def test_json_output(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(cli, "build_subject", lambda n: _FakeSubject([_good_answer()] * 3))
        assert cli.main(_make_args(tmp_path, extra=["--json"])) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["subject"] == "fake"
        assert payload["metrics"]["answer_rate"] == 1.0

    def test_ollama_judge_constructed_with_model(self, tmp_path, monkeypatch):
        seen = {}

        def fake_judge(**kwargs):
            seen.update(kwargs)
            return HeuristicJudge()

        monkeypatch.setattr(cli, "OllamaJudge", fake_judge)
        monkeypatch.setattr(cli, "build_subject", lambda n: _FakeSubject([_good_answer()] * 3))
        assert (
            cli.main(_make_args(tmp_path, extra=["--judge", "ollama", "--model", "qwen3:8b"]))
            == 0
        )
        assert seen["model"] == "qwen3:8b"
        # default model env falls back gracefully when not provided
        monkeypatch.delenv("OLLAMA_MODEL", raising=False)
        assert (
            cli.main(_make_args(tmp_path, extra=["--judge", "ollama"]))
            == 0
        )

    def test_missing_golden_returns_fail(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(cli, "build_subject", lambda n: _FakeSubject([]))
        assert (
            cli.main(
                [
                    "--subject", "alpha-agent",
                    "--golden", str(tmp_path / "missing.json"),
                    "--report-dir", str(tmp_path),
                ]
            )
            == 1
        )
        assert "error: Golden dataset not found" in capsys.readouterr().err

    def test_artifacts_on_stderr(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(cli, "build_subject", lambda n: _FakeSubject([_good_answer()] * 3))
        cli.main(_make_args(tmp_path))
        err = capsys.readouterr().err
        assert "[artifacts]" in err
        assert str(tmp_path) in err

    def test_threshold_env_invalid_returns_fail(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("EVALFORGE_THRESHOLD", "not-a-pair")
        monkeypatch.setattr(cli, "build_subject", lambda n: _FakeSubject([_good_answer()] * 3))
        assert cli.main(_make_args(tmp_path)) == 1
        assert "error:" in capsys.readouterr().err


class TestThresholdConfig:
    def test_as_dict_complete(self):
        data = RegressionThresholds().as_dict()
        assert data["min_answer_rate"] == 0.70
        assert data["min_faithfulness"] == 0.70
        assert data["max_hallucination_rate"] == 0.10
        assert data["warn_margin"] == 0.05