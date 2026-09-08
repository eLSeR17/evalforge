"""Subjects: anything evalforge can ask questions to.

A *subject* is the system under evaluation — an LLM pipeline, an agent, or a
wrapped container. The toolkit only requires it to expose
:meth:`Subject.ask`::

    answer = subject.ask("What is the current price of AAPL?")

:class:`CliSubject` adapts any command-line program into a subject by running
it via ``subprocess`` and capturing stdout. The two portfolio projects are
registered in :data:`SUBJECTS` and adapted this way from *outside* their
containers (``docker exec``) — evalforge never imports their code.

Quoting robustness
------------------
Questions may contain spaces, quotes, or shell metacharacters. The safe
pattern is to pass the question **base64-encoded in an environment variable**
and decode it inside the container::

    docker exec -e Q=<base64> <container> sh -c 'printf %s "$Q" | base64 -d | ...'

Because the base64 alphabet is ``[A-Za-z0-9+/=]`` it can never collide with
shell syntax, so embedding it in a ``sh -c`` string is always safe. The
alternative ``{question}`` placeholder is only for list-form commands where
the question becomes its *own* argv element (no shell involved).
"""

from __future__ import annotations

import base64
import re
import subprocess
from dataclasses import dataclass, field
from typing import Callable, Protocol

from .models import SubjectAnswer

# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class Subject(Protocol):
    """Anything that can be asked a question."""

    name: str

    def ask(self, question: str) -> SubjectAnswer:
        """Answer *question*, returning the answer and any exposed provenance."""
        ...


class SubjectError(RuntimeError):
    """Raised when a subject fails to produce an answer (timeout, crash, ...)."""


# ---------------------------------------------------------------------------
# CLI adapter
# ---------------------------------------------------------------------------


@dataclass
class CliSubject:
    """Adapt a command-line program into a :class:`Subject`.

    Parameters
    ----------
    name:
        Subject name (used in reports).
    command:
        Argv template. Exactly one placeholder per command element:
        ``{question}`` (must be a *whole* argv element) or ``{question_b64}``
        (safe anywhere, intended for ``docker exec -e Q=...`` patterns).
    refusal_pattern:
        Compiled regex matched against stdout to mark an answer as a refusal
        (e.g. ``I DON'T KNOW``, a blocked-request disclaimer, ...).
    parse_sources:
        Optional ``stdout -> list[str]`` extractor for cited/grounding source
        labels.
    enrich_retrieved_ids:
        Optional ``stdout -> list[str]`` extractor for retrieved document ids
        (enables context precision/recall).
    timeout:
        Per-question subprocess timeout in seconds.
    """

    name: str
    command: list[str] = field(default_factory=list)
    refusal_pattern: re.Pattern | None = None
    parse_sources: Callable[[str], list[str]] | None = None
    enrich_retrieved_ids: Callable[[str], list[str]] | None = None
    timeout: float = 120.0

    def __post_init__(self) -> None:
        has_question = any("{question}" in arg for arg in self.command)
        has_question_b64 = any("{question_b64}" in arg for arg in self.command)
        if not has_question and not has_question_b64:
            raise ValueError(
                "CliSubject.command must contain a {question} or {question_b64} placeholder"
            )
        if "{question}" in self.command and "{question_b64}" in self.command:
            raise ValueError("use either {question} or {question_b64}, not both")

    # ------------------------------------------------------------------
    def ask(self, question: str) -> SubjectAnswer:
        """Run the command for *question* and map stdout to an answer."""
        cmd = self._render(self.command, question)
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise SubjectError(
                f"{self.name} timed out after {self.timeout}s answering: {question!r}"
            ) from exc

        stdout = proc.stdout or ""
        sources = self.parse_sources(stdout) if self.parse_sources else []
        retrieved = self.enrich_retrieved_ids(stdout) if self.enrich_retrieved_ids else []
        refused = self._detect_refusal(stdout)
        return SubjectAnswer(
            answer=stdout,
            sources=sources,
            retrieved_doc_ids=retrieved,
            refused=refused,
        )

    # ------------------------------------------------------------------
    def _detect_refusal(self, stdout: str) -> bool:
        """Mark a refusal: empty output or a match of the refusal pattern."""
        if not stdout.strip():
            return True
        if self.refusal_pattern is None:
            return False
        return bool(self.refusal_pattern.search(stdout))

    @staticmethod
    def _render(command: list[str], question: str) -> list[str]:
        """Render the argv template with the question.

        ``{question_b64}`` is replaced by the base64 encoding of the question
        (shell-safe by construction). ``{question}`` must occupy a whole argv
        element; embedding it inside a shell string would break quoting, so
        that misuse raises ``ValueError``.
        """
        rendered: list[str] = []
        for arg in command:
            if "{question_b64}" in arg:
                b64 = base64.b64encode(question.encode("utf-8")).decode("ascii")
                rendered.append(arg.replace("{question_b64}", b64))
            elif "{question}" in arg:
                if arg.strip() != "{question}":
                    raise ValueError(
                        "the {question} placeholder must be a whole argv element "
                        "(use {question_b64} when the question is embedded in a shell string)"
                    )
                rendered.append(question)
            else:
                rendered.append(arg)
        return rendered


# ---------------------------------------------------------------------------
# Registry: the two portfolio subjects
# ---------------------------------------------------------------------------

#: smart-contract-rag refuses with "I DON'T KNOW" (its anti-hallucination
#: output guardrail) and, on refusal, the CLI also prints
#: "(Response was refused pending grounding verification.)".
_SCR_REFUSAL_PATTERN = re.compile(
    r"response was refused pending grounding verification"
    r"|i don'?t know"
    r"|i do not know"
    r"|dont know"
    r"|not enough"
    r"|no relevant sources",
    re.IGNORECASE,
)

#: alpha-agent declines when guardrails block a query ("I'm sorry, but I
#: cannot process this request."), when the query is missing a ticker (it asks
#: for the symbol), or when it cannot complete/find data. The financial
#: disclaimer is appended to *every* answer and is NOT a refusal signal.
_AA_REFUSAL_PATTERN = re.compile(
    r"i'?m sorry, but i cannot process this request"
    r"|please provide (?:a |the |your )?(?:valid )?(?:ticker|symbol|stock)"
    r"|which ticker"
    r"|ticker symbol"
    r"|unable to complete the analysis"
    r"|i don'?t (?:know|have)"
    r"|i do not (?:know|have)"
    r"|cannot (?:predict|provide that|answer that|determine)"
    r"|not a valid ticker"
    r"|no data (?:available|found)"
    r"|could not (?:retrieve|find|get)",
    re.IGNORECASE,
)

#: Source lines emitted by `python -m smart_contract_rag.cli`:
#: `  - aave-v3 (page 41)` (verified against the subject's cli.py).
_SOURCE_LINE_RE = re.compile(
    r"^\s*-\s+([A-Za-z0-9._-]+)\s+\(page\s+(\d+|n/a)\)\s*$", re.MULTILINE
)


def _parse_rag_sources(stdout: str) -> list[str]:
    """Extract source labels like ``"aave-v3 (page 41)"`` from RAG stdout."""
    return [f"{m.group(1)} (page {m.group(2)})" for m in _SOURCE_LINE_RE.finditer(stdout)]


def _parse_rag_retrieved_ids(stdout: str) -> list[str]:
    """Extract the retrieved document ids from RAG stdout (in order)."""
    return [m.group(1) for m in _SOURCE_LINE_RE.finditer(stdout)]


#: Alpha-agent runs inside the `alpha-agent-demo` container with the repo
#: mounted at /repo (created on demand by scripts/run_e2e.py). The question
#: is base64-encoded in `Q`, decoded, piped on stdin, and the wrapper from the
#: subject's README ("Run the agent") is executed with zero nested quoting.
_ALPHA_AGENT_SCRIPT = (
    'printf %s "$Q" | base64 -d | '
    'PYTHONPATH=/repo/src python -c "import sys\n'
    'from alpha_agent import AlphaAgent, GuardedAlphaAgent, OllamaClient\n'
    'from alpha_agent.tools import TOOL_REGISTRY\n'
    'agent = GuardedAlphaAgent(AlphaAgent(llm_client=OllamaClient(), '
    'tools=list(TOOL_REGISTRY.values())))\n'
    'print(agent.analyze(sys.stdin.read()).final_answer)"'
)

#: smart-contract-rag runs in the existing `scr-rag-demo` container (repo at
#: /app). The decoded question is passed as the CLI's positional argument.
_SCR_SCRIPT = 'PYTHONPATH=/app/src python -m smart_contract_rag.cli "$(printf %s "$Q" | base64 -d)"'

#: Registry of available subjects. Each entry is a zero-argument factory for a
#: :class:`CliSubject`; ``build_subject`` resolves names for the CLI.
SUBJECTS: dict[str, Callable[[], CliSubject]] = {
    "alpha-agent": lambda: CliSubject(
        name="alpha-agent",
        command=[
            "docker", "exec", "-e", "Q={question_b64}",
            "alpha-agent-demo", "sh", "-c", _ALPHA_AGENT_SCRIPT,
        ],
        refusal_pattern=_AA_REFUSAL_PATTERN,
        parse_sources=None,
        enrich_retrieved_ids=None,
        timeout=180.0,
    ),
    "smart-contract-rag": lambda: CliSubject(
        name="smart-contract-rag",
        command=[
            "docker", "exec", "-e", "Q={question_b64}",
            "scr-rag-demo", "sh", "-c", _SCR_SCRIPT,
        ],
        refusal_pattern=_SCR_REFUSAL_PATTERN,
        parse_sources=_parse_rag_sources,
        enrich_retrieved_ids=_parse_rag_retrieved_ids,
        timeout=120.0,
    ),
}


def build_subject(name: str) -> CliSubject:
    """Resolve a subject name from :data:`SUBJECTS`.

    Raises ``KeyError`` for unknown names.
    """
    try:
        return SUBJECTS[name]()
    except KeyError:
        raise KeyError(
            f"unknown subject {name!r}; available: {', '.join(sorted(SUBJECTS))}"
        ) from None