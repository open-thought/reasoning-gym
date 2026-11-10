"""Regression tests for numeric substring rewards (issue #566)."""

import pytest

import reasoning_gym
from reasoning_gym.utils import compute_decimal_reward, compute_string_reward


@pytest.mark.parametrize("score", [compute_string_reward, compute_decimal_reward])
@pytest.mark.parametrize(
    "answer",
    [
        "15",
        "-5",
        "5.1",
        "5e2",
        "5 or 6",
        "0 1 2 3 4 5 6 7 8 9",
        "5/2",
        "5%",
        "−5",
        "- 5",
        "5 +",
        "5..0",
        "5e",
        "id5",
        "5kg",
    ],
)
def test_wrong_numbers_and_fragments(score, answer):
    assert score(answer, "5") == 0.0


@pytest.mark.parametrize("score", [compute_string_reward, compute_decimal_reward])
@pytest.mark.parametrize(
    "answer,oracle",
    [
        ("The answer is 5.", "5"),
        ("The answer is 5, obviously.", "5"),
        ("My best-guess is 5", "5"),
        ("Answer: -5", "-5"),
        ("The answer is 5.0.", "5.0"),
        ("The answer is 5e0.", "5e0"),
    ],
)
def test_literal_prose_and_punctuation(score, answer, oracle):
    # The decimal helper retains its historical comma stripping before scoring.
    scored_answer = answer.replace(",", "") if score is compute_decimal_reward else answer
    assert score(answer, oracle) == len(oracle) / len(scored_answer)


@pytest.mark.parametrize("answer,oracle", [("1", "001"), ("101.0", "101"), ("5e0", "5")])
def test_equivalence_is_only_for_standalone_decimal_answers(answer, oracle):
    assert compute_string_reward(oracle, oracle) == 1.0
    assert compute_string_reward(answer, oracle) == 0.0
    assert compute_decimal_reward(answer, oracle) == 1.0
    assert compute_decimal_reward("Answer: " + answer, oracle) == 0.0


def test_decimal_helper_preserves_comma_policy():
    assert compute_decimal_reward("1,00", "100") == 1.0
    assert compute_decimal_reward("1,00", "100", strip_commas=False) == 0.0


@pytest.mark.parametrize("value", ["NaN", "Infinity"])
def test_nonfinite_decimals_receive_zero(value):
    assert compute_decimal_reward(value, value) == 0.0


@pytest.mark.parametrize("oracle", ["001", "cat", "[1, 2]", "1 0\n0 1"])
def test_literal_substring_credit(oracle):
    answer = "Result: " + oracle
    assert compute_string_reward(answer, oracle) == len(oracle) / len(answer)


@pytest.mark.parametrize("name", ["gcd", "chain_sum"])
def test_default_and_decimal_dataset_scoring(name):
    dataset = reasoning_gym.create_dataset(name, size=1, seed=42)
    entry = dataset[0]
    oracle = entry["answer"]
    assert dataset.score_answer(oracle, entry) == 1.0
    assert dataset.score_answer("The answer is " + oracle, entry) > 0.0
    assert dataset.score_answer("1" + oracle, entry) == 0.0
    assert dataset.score_answer(oracle + " or 123456789", entry) == 0.0
