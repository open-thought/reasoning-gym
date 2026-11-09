"""Regression contracts for scalar numeric scoring (issue #566)."""

import pytest

import reasoning_gym
from reasoning_gym.dataset import ProceduralDataset
from reasoning_gym.utils import compute_decimal_reward


@pytest.mark.parametrize(
    "answer,oracle",
    [
        ("5", "5"),
        (" +005.00 ", "5"),
        ("5e0", "5"),
        ("1,000", "1000"),
        ("+0001,000.00", "1000"),
        (".5", "0.5"),
        ("-0", "0"),
        ("-5", "-5"),
    ],
)
def test_equivalent_standalone_numbers(answer, oracle):
    assert compute_decimal_reward(answer, oracle) == 1.0


@pytest.mark.parametrize(
    "answer",
    [
        "15",
        "-5",
        "5.1",
        "50",
        "5e2",
        "5 or 6",
        "5 5",
        "0 1 2 3 4 5 6 7 8 9",
        "5/2",
        "5%",
        "5+1",
        "5*1",
        "5^1",
        "−5",
        "- 5",
        "5+",
        "5-",
        "x=5",
        "1,5",
        "5,00",
        "5..0",
        "5e",
        "id5",
        "5kg",
        "NaN",
        "Infinity",
        "nan or 5",
        "inf or 5",
        "",
        None,
    ],
)
def test_invalid_or_wrong_numbers_receive_zero(answer):
    assert compute_decimal_reward(answer, "5") == 0.0


@pytest.mark.parametrize("answer", ["The answer is 5.", "Answer: -5", "Result: 5e0", "5 + x"])
def test_single_number_in_prose(answer):
    oracle = "-5" if "-5" in answer else "5"
    assert 0.0 < compute_decimal_reward(answer, oracle) < 1.0


@pytest.mark.parametrize("oracle", ["nan", "inf", "Infinity", "1,00", "", "five"])
def test_invalid_oracle(oracle):
    assert compute_decimal_reward("The answer is 5", oracle) == 0.0


def test_commas_can_be_disabled():
    assert compute_decimal_reward("1,000", "1000", strip_commas=False) == 0.0
    assert compute_decimal_reward("Result: 1,000", "1000", strip_commas=False) == 0.0
    assert compute_decimal_reward("Result: 1,000", "1000") > 0.0


NUMERIC_DATASETS = [
    "basic_arithmetic",
    "gcd",
    "lcm",
    "count_bits",
    "leg_counting",
    "aiw",
    "number_sequence",
    "largest_island",
    "count_primes",
    "binary_alternation",
    "letter_counting",
    "rotten_oranges",
    "maze",
    "simple_equations",
    "simple_geometry",
    "products",
    "chain_sum",
]


@pytest.mark.parametrize("name", NUMERIC_DATASETS)
def test_numeric_dataset_integration(name):
    dataset = reasoning_gym.create_dataset(name, size=3, seed=42)
    for entry in dataset:
        oracle = entry["answer"]
        assert dataset.score_answer(oracle, entry) == 1.0
        assert dataset.score_answer("The answer is " + oracle, entry) > 0.0
        assert dataset.score_answer("1" + oracle, entry) == 0.0
        assert dataset.score_answer(oracle + " or 123456789", entry) == 0.0
        assert dataset.score_answer(None, entry) == 0.0


class TextDataset(ProceduralDataset):
    def __getitem__(self, idx):
        return {"answer": "001", "metadata": {}, "question": "Return the string"}


def test_numeric_policy_is_explicit():
    dataset = TextDataset(config=None, size=1)
    entry = dataset[0]
    assert dataset.score_answer("001", entry) == 1.0
    assert dataset.score_answer("1", entry) == 0.0
    assert dataset.score_answer("1001", entry) == 0.75


@pytest.mark.parametrize("name", ["base_conversion", "bf", "number_filtering", "rotate_matrix", "syllogism"])
def test_representation_sensitive_datasets_keep_text_policy(name):
    assert reasoning_gym.create_dataset(name, size=1, seed=42).numeric_answer is False
