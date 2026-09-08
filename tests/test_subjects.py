"""Unit tests for subject adapters (evalforge.subjects).

All tests are hermetic: ``subprocess.run`` is monkeypatched, so nothing ever
touches Docker or the real sibling containers.
"""

import base64
import subprocess
from dataclasses import dataclass

import pytest

from evalforge.models import SubjectAnswer
from evalforge.subjects import (
    SUBJECTS,
    CliSubject,
    SubjectError,
    _parse_rag_retrieved_ids,
    _parse_rag_sources,
    build_subject,
)

_RAG_STDOUT = (
    "Subject: reentrancy\n\n"
    "Answer: the fix is to add a reentrancy guard.\n\n"
    "Sources:\n"
    "  - aave-v3 (page 41)\n"
    "  - balancer-managedpool (page 5)\n"
)


@dataclass
class _Proc:
    stdout: str = ""
    returncode: int = 0
    stderr: str = ""


def _scr_subject() -> CliSubject:
    return build_subject("smart-contract-rag")


class TestCliSubjectValidation:
    def test_requires_placeholder(self):
        with pytest.raises(ValueError, match="placeholder"):
            CliSubject(name="x", command=["python", "-c", "print('hi')"])

    def test_both_placeholders_rejected(self):
        with pytest.raises(ValueError, match="either"):
            CliSubject(
                name="x",
                command=["python", "{question}", "{question_b64}"],
            )

    def test_question_placeholder_embedded_raises_on_ask(self):
        subject = CliSubject(
            name="x",
            command=["sh", "-c", 'echo "{question}"'],
        )
        with pytest.raises(ValueError, match="whole argv element"):
            subject.ask("hello world")

    def test_question_placeholder_whole_argv(self):
        subject = CliSubject(name="x", command=["python", "-c", "{question}"])
        cmd = subject._render(subject.command, "hello world")
        assert cmd == ["python", "-c", "hello world"]


class TestBase64Rendering:
    """The {question_b64} placeholder must survive any user input."""

    @pytest.mark.parametrize(
        "question",
        [
            "plain question",
            "quotes ' and \" inside",
            "shell $HOME metachar$ & backtick `",
            "newline\nquestion",
            "unicode: ¿qué es? æøå",
        ],
    )
    def test_roundtrip(self, question):
        subject = CliSubject(name="x", command=["prog", "-p", "Q={question_b64}", "-r"])
        cmd = subject._render(subject.command, question)
        encoded = next(a for a in cmd if a.startswith("Q="))[2:]
        assert base64.b64decode(encoded).decode("utf-8") == question

    def test_in_env_variable_position(self):
        subject = CliSubject(name="x", command=["docker", "exec", "-e", "Q={question_b64}", "ctr", "sh", "-c", "echo ok"])
        cmd = subject._render(subject.command, "hello world")
        assert cmd[3] == "Q=" + base64.b64encode(b"hello world").decode("ascii")


class TestAskParsing:
    def test_parses_rag_stdout(self, monkeypatch):
        def fake_run(cmd, **kwargs):
            assert cmd[0] == "docker"
            assert kwargs.get("capture_output") is True
            return _Proc(stdout=_RAG_STDOUT)

        monkeypatch.setattr(subprocess, "run", fake_run)
        subject = _scr_subject()
        answer = subject.ask("What fixes a reentrancy attack?")
        assert isinstance(answer, SubjectAnswer)
        assert answer.answer == _RAG_STDOUT
        assert answer.sources == ["aave-v3 (page 41)", "balancer-managedpool (page 5)"]
        assert answer.retrieved_doc_ids == ["aave-v3", "balancer-managedpool"]
        assert answer.refused is False

    def test_refusal_pattern_marks_refusal(self, monkeypatch):
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda cmd, **kwargs: _Proc(stdout="I DON'T KNOW anything about this."),
        )
        subject = _scr_subject()
        assert subject.ask("What is the answer?").refused is True

    def test_refused_pending_grounding_marks_refusal(self, monkeypatch):
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda cmd, **kwargs: _Proc(
                stdout="(Response was refused pending grounding verification.)"
            ),
        )
        subject = _scr_subject()
        assert subject.ask("question").refused is True

    def test_empty_stdout_is_refusal(self, monkeypatch):
        monkeypatch.setattr(subprocess, "run", lambda cmd, **kwargs: _Proc(stdout=""))
        subject = _scr_subject()
        answer = subject.ask("question")
        assert answer.refused is True
        assert answer.sources == []

    def test_non_matching_stdout_is_not_refusal(self, monkeypatch):
        monkeypatch.setattr(
            subprocess, "run", lambda cmd, **kwargs: _Proc(stdout="a normal answer here")
        )
        subject = _scr_subject()
        assert subject.ask("question").refused is False

    def test_timeout_raises_subject_error(self, monkeypatch):
        def boom(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=120)

        monkeypatch.setattr(subprocess, "run", boom)
        subject = CliSubject(
            name="slow",
            command=["python", "{question}"],
            timeout=120.0,
        )
        with pytest.raises(SubjectError, match="timed out"):
            subject.ask("question")


class TestRagSourceParsing:
    def test_parse_sources_keeps_page(self):
        assert _parse_rag_sources("  - aave-v3 (page 41)\n") == ["aave-v3 (page 41)"]

    def test_parse_sources_handles_page_n_a(self):
        assert _parse_rag_sources("  - beanstalk-security (page n/a)\n") == [
            "beanstalk-security (page n/a)"
        ]

    def test_parse_sources_multiple_lines(self):
        out = _parse_rag_sources("  - aave-v3 (page 41)\n  - b-2 (page 5)\n")
        assert len(out) == 2

    def test_parse_sources_ignores_non_source_lines(self):
        out = _parse_rag_sources("Answer text here.\nSources:\n  - aave-v3 (page 41)\n")
        assert out == ["aave-v3 (page 41)"]

    def test_parse_sources_empty(self):
        assert _parse_rag_sources("just prose") == []

    def test_retrieved_ids_in_order(self):
        assert _parse_rag_retrieved_ids("  - b-2 (page 1)\n  - aave-v3 (page 9)\n") == [
            "b-2",
            "aave-v3",
        ]


class TestRegistry:
    def test_resolves_alpha_agent(self):
        subject = build_subject("alpha-agent")
        assert subject.name == "alpha-agent"
        assert subject.command[0] == "docker"
        assert subject.timeout == 180.0

    def test_resolves_smart_contract_rag(self):
        subject = build_subject("smart-contract-rag")
        assert subject.name == "smart-contract-rag"
        assert subject.command[3] == "Q={question_b64}"
        assert subject.timeout == 120.0

    def test_unknown_subject_raises(self):
        with pytest.raises(KeyError, match="unknown subject"):
            build_subject("nope")

    def test_subject_names(self):
        assert sorted(SUBJECTS) == ["alpha-agent", "smart-contract-rag"]

    def test_alpha_question_never_quoted_in_shell(self):
        # The static wrapper contains no {question} placeholder: the question
        # only travels as base64 in the env var. This is the zero-nested-
        # quoting guarantee.
        command = build_subject("alpha-agent").command
        assert "{question_b64}" in command[3]
        assert all("{question}" not in arg for arg in command)