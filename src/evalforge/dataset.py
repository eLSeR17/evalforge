"""Golden dataset loading and validation.

A *golden dataset* is a curated, versioned set of
:class:`~evalforge.models.EvalCase` objects representing the ground truth an
eval run measures a subject against. It is the same concept as the golden
sets shipped in the sibling ``smart-contract-rag`` project, but generalised:

- no domain-specific schema terms (``relevant_doc_id`` becomes ``doc_ids``;
  ``expect_answer`` becomes ``refuse``),
- validation reports *paths* to the offending field (``cases[3].doc_ids``)
  so a malformed dataset fails fast in CI with an actionable message.

JSON schema
-----------
A dataset file is a JSON array of objects::

    [
      {
        "id": "sr-001",
        "topic": "reentrancy",
        "question": "What is a reentrancy attack ...?",
        "expected_keywords": ["external", "call", "state"],
        "refuse": false,
        "doc_ids": ["aave-v3"]
      }
    ]

``expected_keywords`` must be non-empty for answerable cases (``refuse``
false) and empty for refusal cases. Unknown extra keys are ignored so the
schema can evolve without breaking older datasets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .models import EvalCase


def default_golden_filename(subject: str) -> str:
    """Return the conventional golden dataset filename for a *subject*.

    Convention: subjects are registered kebab-case (``alpha-agent``,
    ``smart-contract-rag``) while package and data file names are snake_case,
    so the default filename mirrors the subject with ``-`` -> ``_``:
    ``alpha-agent`` -> ``alpha_agent.json``. ``--golden`` overrides it.
    """
    return f"{subject.replace('-', '_')}.json"


@dataclass(frozen=True)
class DatasetIssue:
    """A single schema violation, located by JSON path."""

    path: str
    message: str


@dataclass(frozen=True)
class DatasetValidation:
    """Result of validating a dataset's schema."""

    issues: list[DatasetIssue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.issues

    def __bool__(self) -> bool:
        return self.valid


@dataclass
class EvalDataset:
    """A validated collection of :class:`EvalCase` objects."""

    cases: list[EvalCase]
    source: str = "<memory>"

    # ------------------------------------------------------------------
    @classmethod
    def from_json(cls, path: str | Path) -> "EvalDataset":
        """Load and validate a golden dataset from a JSON file.

        Raises ``FileNotFoundError`` for a missing file, ``ValueError`` for
        invalid JSON or schema violations.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Golden dataset not found: {p}")
        try:
            with p.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in golden dataset {p}: {exc}") from exc

        if not isinstance(raw, list):
            raise ValueError("Golden dataset JSON must be a list of case objects.")

        cases = [_case_from_dict(i, entry) for i, entry in enumerate(raw)]
        dataset = cls(cases=cases, source=str(p))
        report = dataset.validate()
        if not report.valid:
            raise ValueError(_format_issues(report))
        return dataset

    @classmethod
    def from_cases(cls, cases: list[EvalCase] | list[dict]) -> "EvalDataset":
        """Build a dataset from in-memory cases or plain dicts (tests)."""
        if cases and isinstance(cases[0], dict):
            return cls(cases=[_case_from_dict(i, c) for i, c in enumerate(cases)])
        return cls(cases=[c for c in cases if isinstance(c, EvalCase)])

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self) -> DatasetValidation:
        """Return a report of every schema violation (never raises)."""
        issues: list[DatasetIssue] = []
        seen_ids: set[str] = set()

        for i, case in enumerate(self.cases):
            issues.extend(_validate_case(i, case, seen_ids))

        if not self.cases:
            issues.append(DatasetIssue("$", "dataset must contain at least one case"))

        return DatasetValidation(issues=issues)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self) -> Iterator[EvalCase]:
        return iter(self.cases)

    def __getitem__(self, index: int) -> EvalCase:
        return self.cases[index]


def _case_from_dict(index: int, entry: Any) -> EvalCase:
    """Coerce one raw dict entry into an ``EvalCase`` (types validated later)."""
    if not isinstance(entry, dict):
        raise ValueError(f"cases[{index}] must be an object, got {type(entry).__name__}")
    return EvalCase(
        id=str(entry.get("id", "")),
        topic=str(entry.get("topic", "")),
        question=str(entry.get("question", "")),
        expected_keywords=list(entry.get("expected_keywords", []) or []),
        refuse=bool(entry.get("refuse", False)),
        doc_ids=list(entry.get("doc_ids", []) or []),
    )


def _validate_case(index: int, case: EvalCase, seen_ids: set[str]) -> list[DatasetIssue]:
    """Validate one case, returning its issues with JSON paths."""
    issues: list[DatasetIssue] = []
    base = f"cases[{index}]"

    if not case.id.strip():
        issues.append(DatasetIssue(f"{base}.id", "missing or empty"))
    elif case.id in seen_ids:
        issues.append(DatasetIssue(f"{base}.id", f"duplicate id {case.id!r}"))
    seen_ids.add(case.id)

    if not case.question.strip():
        issues.append(DatasetIssue(f"{base}.question", "missing or empty"))

    if not case.topic.strip():
        issues.append(DatasetIssue(f"{base}.topic", "missing or empty"))

    if case.refuse:
        if case.expected_keywords:
            issues.append(
                DatasetIssue(
                    f"{base}.expected_keywords",
                    "must be empty for refusal cases (traps have no expected content)",
                )
            )
    else:
        if not case.expected_keywords:
            issues.append(
                DatasetIssue(
                    f"{base}.expected_keywords",
                    "must be non-empty for answerable cases",
                )
            )
        elif not all(isinstance(k, str) and k.strip() for k in case.expected_keywords):
            issues.append(
                DatasetIssue(
                    f"{base}.expected_keywords",
                    "all entries must be non-empty strings",
                )
            )

    if any(not isinstance(d, str) or not d.strip() for d in case.doc_ids):
        issues.append(DatasetIssue(f"{base}.doc_ids", "all entries must be non-empty strings"))

    return issues


def _format_issues(report: DatasetValidation) -> str:
    lines = ["Golden dataset failed validation:"]
    for issue in report.issues:
        lines.append(f"  - {issue.path}: {issue.message}")
    return "\n".join(lines)