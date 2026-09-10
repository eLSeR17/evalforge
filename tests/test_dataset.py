"""Unit tests for golden dataset loading and validation (evalforge.dataset)."""

import json
from pathlib import Path

import pytest

from evalforge.dataset import DatasetIssue, EvalDataset, default_golden_filename
from evalforge.models import EvalCase

_GOLDEN_DIR = Path(__file__).resolve().parent.parent / "data" / "golden"


def _valid_case(cid: str = "c1", **overrides) -> dict:
    case = {
        "id": cid,
        "topic": "reentrancy",
        "question": "What is the fix for a reentrancy attack?",
        "expected_keywords": ["reentrancy", "guard"],
        "refuse": False,
        "doc_ids": ["aave-v3"],
    }
    case.update(overrides)
    return case


class TestGoldenFiles:
    """The shipped demo datasets must load and validate cleanly."""

    def test_alpha_agent_golden_valid(self):
        dataset = EvalDataset.from_json(_GOLDEN_DIR / "alpha_agent.json")
        assert len(dataset) == 10
        ids = {c.id for c in dataset}
        assert {f"fa-{i:03d}" for i in range(1, 11)} == ids

    def test_smart_contract_rag_golden_valid(self):
        dataset = EvalDataset.from_json(_GOLDEN_DIR / "smart_contract_rag.json")
        assert len(dataset) == 12
        ids = {c.id for c in dataset}
        assert {f"sr-{i:03d}" for i in range(1, 13)} == ids

    def test_golden_refusal_cases_have_no_keywords(self):
        for name in ("alpha_agent.json", "smart_contract_rag.json"):
            dataset = EvalDataset.from_json(_GOLDEN_DIR / name)
            for case in dataset:
                if case.refuse:
                    assert case.expected_keywords == []


class TestDefaultGoldenFilename:
    """Naming convention: subject kebab-case -> golden file snake_case."""

    def test_kebab_case_subjects_map_to_snake_case(self):
        assert default_golden_filename("alpha-agent") == "alpha_agent.json"
        assert default_golden_filename("smart-contract-rag") == "smart_contract_rag.json"

    def test_subject_without_dash_is_unchanged(self):
        assert default_golden_filename("foo") == "foo.json"
        assert default_golden_filename("sc_rag") == "sc_rag.json"

    def test_never_emits_a_hyphen(self):
        for subject in ("alpha-agent", "smart-contract-rag", "a-b-c"):
            assert "-" not in default_golden_filename(subject)

    def test_matches_the_shipped_golden_files(self):
        # Both real demo datasets must be reachable via the convention.
        assert (_GOLDEN_DIR / default_golden_filename("alpha-agent")).exists()
        assert (_GOLDEN_DIR / default_golden_filename("smart-contract-rag")).exists()

    def test_usable_with_a_golden_dir_path(self):
        assert (Path("data") / "golden" / default_golden_filename("alpha-agent")) == Path(
            "data/golden/alpha_agent.json"
        )


class TestValidation:
    def test_valid_case_validates(self):
        dataset = EvalDataset.from_cases([_valid_case()])
        assert dataset.validate().valid

    def test_duplicate_id_reported(self):
        dataset = EvalDataset.from_cases([_valid_case(), _valid_case()])
        report = dataset.validate()
        assert not report.valid
        issue = report.issues[0]
        assert issue.path == "cases[1].id"
        assert "duplicate" in issue.message

    def test_empty_question_reported(self):
        dataset = EvalDataset.from_cases([_valid_case(question=" ")])
        report = dataset.validate()
        paths = {i.path for i in report.issues}
        assert "cases[0].question" in paths

    def test_empty_topic_reported(self):
        dataset = EvalDataset.from_cases([_valid_case(topic="")])
        report = dataset.validate()
        paths = {i.path for i in report.issues}
        assert "cases[0].topic" in paths

    def test_answerable_requires_keywords(self):
        dataset = EvalDataset.from_cases([_valid_case(expected_keywords=[])])
        report = dataset.validate()
        paths = {i.path for i in report.issues}
        assert "cases[0].expected_keywords" in paths

    def test_refusal_case_must_not_have_keywords(self):
        dataset = EvalDataset.from_cases(
            [_valid_case(refuse=True, doc_ids=[], expected_keywords=["x"])]
        )
        report = dataset.validate()
        paths = {i.path for i in report.issues}
        assert "cases[0].expected_keywords" in paths

    def test_non_string_keyword_reported(self):
        dataset = EvalDataset.from_cases([_valid_case(expected_keywords=["ok", 42])])
        report = dataset.validate()
        paths = {i.path for i in report.issues}
        assert "cases[0].expected_keywords" in paths

    def test_empty_doc_id_entry_reported(self):
        dataset = EvalDataset.from_cases([_valid_case(doc_ids=["aave-v3", " "])])
        report = dataset.validate()
        paths = {i.path for i in report.issues}
        assert "cases[0].doc_ids" in paths

    def test_empty_dataset_reported(self):
        dataset = EvalDataset.from_cases([])
        report = dataset.validate()
        assert not report.valid
        assert report.issues[-1].path == "$"

    def test_validate_never_raises(self):
        dataset = EvalDataset.from_cases([_valid_case(), {"id": "x"}])
        report = dataset.validate()  # must not raise even for garbage
        assert isinstance(report.issues, list)
        assert all(isinstance(i, DatasetIssue) for i in report.issues)

    def test_valid_issue_negative_path(self):
        dataset = EvalDataset.from_cases([_valid_case(question="")])
        report = dataset.validate()
        assert not bool(report)  # DatasetValidation.__bool__


class TestFromJson:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            EvalDataset.from_json(tmp_path / "nope.json")

    def test_invalid_json_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid JSON"):
            EvalDataset.from_json(path)

    def test_non_list_raises(self, tmp_path):
        path = tmp_path / "obj.json"
        path.write_text('{"id": "c1"}', encoding="utf-8")
        with pytest.raises(TypeError, match="must be a list"):
            EvalDataset.from_json(path)

    def test_entry_not_dict_raises(self, tmp_path):
        path = tmp_path / "scalar.json"
        path.write_text('[42]', encoding="utf-8")
        with pytest.raises(TypeError, match="must be an object"):
            EvalDataset.from_json(path)

    def test_schema_violation_raises_with_paths(self, tmp_path):
        path = tmp_path / "violation.json"
        path.write_text(json.dumps([_valid_case(question="")]), encoding="utf-8")
        with pytest.raises(ValueError, match=r"cases\[0\]\.question"):
            EvalDataset.from_json(path)

    def test_roundtrip_valid_file(self, tmp_path):
        path = tmp_path / "ok.json"
        path.write_text(json.dumps([_valid_case()]), encoding="utf-8")
        dataset = EvalDataset.from_json(path)
        assert dataset.source == str(path)
        assert len(dataset) == 1
        assert isinstance(dataset[0], EvalCase)

    def test_unknown_extra_keys_ignored(self):
        case = _valid_case()
        case["future_field"] = "ignored"
        dataset = EvalDataset.from_cases([case])
        assert dataset.validate().valid


class TestDatasetContainer:
    def test_len_iter_getitem(self):
        dataset = EvalDataset.from_cases([_valid_case("a"), _valid_case("b")])
        assert len(dataset) == 2
        assert [c.id for c in dataset] == ["a", "b"]
        assert dataset[1].id == "b"

    def test_from_cases_accepts_instances_and_dicts(self):
        dataset = EvalDataset.from_cases([EvalCase(id="a", topic="t", question="q")])
        assert len(dataset) == 1