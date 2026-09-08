"""Unit tests for the deterministic lexical metrics (evalforge.metrics).

Every expected value in this file is hand-computed from the documented
formulas — these are the ground truth the rest of the toolkit builds on.
"""

import pytest

from evalforge.metrics import (
    METRIC_NAMES,
    aggregate,
    answer_relevance,
    citation_accuracy,
    context_precision,
    context_recall,
    correct_refusal,
    extract_doc_id,
    faithfulness,
    hallucination,
    is_refusal,
    tokens,
    word_tokens,
)


class TestTokenization:
    def test_tokens_lowercase_and_split(self):
        assert tokens("Reentrancy attack mitigation") == {"reentrancy", "attack", "mitigation"}

    def test_tokens_keep_numbers(self):
        assert tokens("version 41") == {"version", "41"}

    def test_tokens_drop_stopwords(self):
        assert tokens("the of and") == set()

    def test_tokens_short_words_dropped(self):
        assert "or" not in tokens("or or")

    def test_tokens_keep_dotted_and_underscored_ids(self):
        assert "aave.v3_pool" in tokens("aave.v3_pool")

    def test_word_tokens_keep_stopwords(self):
        assert word_tokens("the answer") == {"the", "answer"}


class TestIsRefusal:
    def test_empty_is_refusal(self):
        assert is_refusal("")
        assert is_refusal("   ")

    def test_known_signals(self):
        for text in (
            "I DON'T KNOW",
            "i do not know the answer",
            "dont know",
            "not enough information",
            "no relevant sources were found",
            "response was refused pending grounding verification",
            "I'm sorry, but I cannot process this request",
        ):
            assert is_refusal(text), text

    def test_normal_answers_not_refusals(self):
        assert not is_refusal("The answer is 42 because of reentrancy.")
        assert not is_refusal("reentrancy attack mitigation")


class TestFaithfulness:
    def test_refusal_vacuously_faithful(self):
        assert faithfulness("I don't know", ref_context="reentrancy guard") == 1.0

    def test_empty_answer_faithful(self):
        assert faithfulness("", ref_context="reentrancy") == 1.0

    def test_no_reference_context_is_none(self):
        assert faithfulness("any answer", ref_context="") is None
        assert faithfulness("any answer", ref_context=None) is None

    def test_fully_contained_answer(self):
        result = faithfulness(
            "reentrancy guard",
            ref_context="reentrancy guard mitigation",
        )
        assert result == 1.0  # 2/2 answer terms contained

    def test_partially_contained_answer(self):
        result = faithfulness(
            "reentrancy mitigation oracle",
            ref_context="reentrancy guard mitigation",
        )
        # answer terms: reentrancy, mitigation, oracle -> 2/3 contained
        assert result == pytest.approx(2 / 3)

    def test_unrelated_answer(self):
        result = faithfulness(
            "spaghetti cooking time",
            ref_context="reentrancy guard",
        )
        assert result == 0.0


class TestAnswerRelevance:
    def test_perfect_keyword_overlap(self):
        result = answer_relevance(
            "the reentrancy guard", expected_keywords=["reentrancy", "guard"]
        )
        assert result == 1.0

    def test_partial_overlap(self):
        result = answer_relevance(
            "only reentrancy here", expected_keywords=["reentrancy", "guard"]
        )
        assert result == 0.5

    def test_refusal_is_neutral(self):
        assert answer_relevance("I don't know", expected_keywords=["reentrancy"]) == 0.0

    def test_no_keywords_no_truth_is_zero(self):
        assert answer_relevance("anything at all") == 0.0

    def test_keywords_win_over_ground_truth(self):
        result = answer_relevance(
            "reentrancy",
            expected_keywords=["reentrancy"],
            ground_truth="completely different truth",
        )
        assert result == 1.0

    def test_ground_truth_fallback(self):
        result = answer_relevance("reentrancy guard", ground_truth="reentrancy guard fix")
        assert result == pytest.approx(2 / 3)

    def test_multiword_keyword_tokens(self):
        result = answer_relevance(
            "checks effects interactions",
            expected_keywords=["checks-effects-interactions"],
        )
        assert result == 1.0


class TestExtractDocId:
    def test_label_with_page(self):
        assert extract_doc_id("aave-v3 (page 41)") == "aave-v3"

    def test_bare_id(self):
        assert extract_doc_id("balancer-managedpool") == "balancer-managedpool"

    def test_leading_whitespace(self):
        assert extract_doc_id("  aave-v3 (page 1)") == "aave-v3"


class TestCitationAccuracy:
    def test_no_expected_docs_is_none(self):
        assert citation_accuracy(["aave-v3 (page 1)"], [], refused=False) is None

    def test_refusal_is_none(self):
        assert citation_accuracy(["aave-v3 (page 1)"], ["aave-v3"], refused=True) is None

    def test_hit_scores_one(self):
        assert (
            citation_accuracy(
                ["aave-v3 (page 41)", "other (page 2)"],
                ["aave-v3"],
                refused=False,
            )
            == 1.0
        )

    def test_miss_scores_zero(self):
        assert (
            citation_accuracy(["unrelated-1 (page 3)"], ["aave-v3"], refused=False)
            == 0.0
        )

    def test_no_sources_scores_zero_for_answer(self):
        assert citation_accuracy([], ["aave-v3"], refused=False) == 0.0


class TestContextPrecision:
    def test_no_retrieval_is_none(self):
        assert context_precision([], ["aave-v3"]) is None

    def test_no_expected_docs_is_none(self):
        assert context_precision(["aave-v3"], []) is None

    def test_half_relevant(self):
        result = context_precision(["aave-v3", "other"], ["aave-v3"])
        assert result == 0.5

    def test_all_relevant(self):
        assert context_precision(["aave-v3", "b-2"], ["aave-v3", "b-2"]) == 1.0


class TestContextRecall:
    def test_no_retrieval_is_none(self):
        assert context_recall([], ["aave-v3"]) is None

    def test_partial_recall(self):
        result = context_recall(["aave-v3"], ["aave-v3", "b-2"])
        assert result == 0.5

    def test_full_recall(self):
        assert context_recall(["b-2", "aave-v3"], ["aave-v3", "b-2"]) == 1.0


class TestCorrectRefusal:
    def test_answered_answerable_is_correct(self):
        assert correct_refusal(refused=False, expected_refusal=False) is True

    def test_refused_answerable_is_wrong(self):
        assert correct_refusal(refused=True, expected_refusal=False) is False

    def test_refused_trap_is_correct(self):
        assert correct_refusal(refused=True, expected_refusal=True) is True

    def test_answered_trap_is_wrong(self):
        assert correct_refusal(refused=False, expected_refusal=True) is False


class TestHallucination:
    def test_refusal_is_not_hallucination(self):
        assert hallucination(refused=True, expected_refusal=True, has_evidence=False) is False
        assert hallucination(refused=True, expected_refusal=False, has_evidence=False) is False

    def test_answered_trap_is_fabrication(self):
        assert hallucination(refused=False, expected_refusal=True, has_evidence=False) is True
        assert hallucination(refused=False, expected_refusal=True, has_evidence=True) is True

    def test_answered_without_evidence_unmeasurable(self):
        assert hallucination(refused=False, expected_refusal=False, has_evidence=False) is None

    def test_answered_with_evidence_ok(self):
        assert hallucination(refused=False, expected_refusal=False, has_evidence=True) is False


class TestAggregate:
    def test_hand_computed_aggregate(self):
        case_metrics = [
            # answered answerable cases
            {
                "answered": True, "refused": False,
                "faith": 0.8, "rel": 0.6, "cit": None, "p_at_k": None, "rec": None,
                "refusal_correct": True, "hallu": None, "expected_refusal": False,
            },
            {
                "answered": True, "refused": False,
                "faith": 0.4, "rel": 0.2, "cit": None, "p_at_k": None, "rec": None,
                "refusal_correct": True, "hallu": None, "expected_refusal": False,
            },
            # refusal cases
            {
                "answered": False, "refused": True,
                "faith": 1.0, "rel": 0.0, "cit": None, "p_at_k": None, "rec": None,
                "refusal_correct": True, "hallu": False, "expected_refusal": True,
            },
            {
                "answered": False, "refused": True,
                "faith": 1.0, "rel": 0.0, "cit": None, "p_at_k": None, "rec": None,
                "refusal_correct": False, "hallu": False, "expected_refusal": True,
            },
        ]
        out = aggregate(case_metrics)
        assert out["faithfulness"] == pytest.approx(0.6)   # mean(0.8, 0.4)
        assert out["answer_relevance"] == pytest.approx(0.4)  # mean(0.6, 0.2)
        assert out["citation_accuracy"] is None  # nothing measurable
        assert out["context_precision"] is None
        assert out["context_recall"] is None
        assert out["answer_rate"] == 1.0  # both answerable answered
        assert out["correct_refusal_rate"] == pytest.approx(0.75)  # 3/4
        assert out["hallucination_rate"] == 0.0  # mean(False, False)

    def test_all_none_when_nothing_measurable(self):
        out = aggregate(
            [{"answered": False, "refused": True, "faith": None, "rel": None,
              "cit": None, "p_at_k": None, "rec": None, "refusal_correct": True,
              "hallu": False, "expected_refusal": True}]
        )
        # Only refusal cases: faith/rel/cit/context are None; answer_rate has
        # no answerable case to average -> None (not a vacuous 1.0).
        assert out["faithfulness"] is None
        assert out["answer_relevance"] is None
        assert out["answer_rate"] is None
        assert out["correct_refusal_rate"] == 1.0
        assert out["hallucination_rate"] == 0.0

    def test_metric_names_are_stable(self):
        assert METRIC_NAMES == (
            "faithfulness",
            "answer_relevance",
            "citation_accuracy",
            "context_precision",
            "context_recall",
            "answer_rate",
            "correct_refusal_rate",
            "hallucination_rate",
        )