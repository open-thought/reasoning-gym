"""Tests for Power Function questions generation"""

from decimal import Decimal

import pytest

from reasoning_gym.arithmetic import PowerFunctionConfig, PowerFunctionDataset


def test_power_function_dataset_deterministic():
    """Test that dataset generates same items with same seed"""
    config = PowerFunctionConfig(seed=42, size=10)
    dataset1 = PowerFunctionDataset(config)
    dataset2 = PowerFunctionDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_power_function_dataset_items():
    """Test basic properties of generated items"""
    config = PowerFunctionConfig(min_base=-100, max_base=-100, min_exponent=-10, max_exponent=10, size=10, seed=42)
    dataset = PowerFunctionDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        # Check item structure
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata
        assert "base" in item["metadata"]
        assert "exponent" in item["metadata"]

        base = item["metadata"]["base"]
        exponent = item["metadata"]["exponent"]
        solution = item["metadata"]["solution"]

        # Verify values
        assert config.min_base <= base <= config.max_base
        assert config.min_exponent <= exponent <= config.max_exponent
        assert solution == pow(base, exponent)


def test_power_function_dataset_iteration():
    """Test that iteration respects dataset size"""
    config = PowerFunctionConfig(size=5, seed=42)
    dataset = PowerFunctionDataset(config)

    items = list(dataset)
    assert len(items) == config.size

    # Test multiple iterations yield same items
    assert items == list(dataset)


def test_power_function_score_function():
    """Test score function"""
    config = PowerFunctionConfig(seed=42)
    dataset = PowerFunctionDataset(config)

    for item in dataset:
        answer = item["answer"]
        assert dataset.score_answer(answer, item) == 1.0


def test_power_function_curriculum():
    """Test PowerFunctionCurriculum configuration generation and attribute manipulation"""
    from reasoning_gym.arithmetic import PowerFunctionCurriculum

    curriculum = PowerFunctionCurriculum()

    base_value = {"size": 150, "seed": 1}

    base_cfg = curriculum.generate_configuration(base_value)
    assert base_cfg.seed == 1
    assert base_cfg.size == 150
    assert base_cfg.min_exponent == 2 and base_cfg.max_exponent == 2

    # Test incrementing attribute levels for exponent & base attributes
    curriculum.increment_attr_level("exponent")

    increased_cfg = curriculum.generate_configuration(base_value)
    assert increased_cfg.min_exponent == 2 and increased_cfg.max_exponent == 4


# Test score_answer function with various answers
def test_power_function_score_answer_for_edge_cases():
    """Test score_answer function for edge cases"""
    config = PowerFunctionConfig(seed=42)
    dataset = PowerFunctionDataset(config)

    # Case 1: Match with trailing zeros
    item = dataset[0].copy()
    user_answer = "1.000e+00"
    # Let's change the oracle answer for edge case testing
    item["answer"] = "1.0"
    score = dataset.score_answer(user_answer, item)
    assert score == 1.0, f"Expected score 1.0, got {score}"

    # Case 2: Rounding up at edge of significant figures
    item = dataset[0].copy()
    item["answer"] = str(Decimal("0.9999") ** 1)  # Close to 1.000
    user_answer = "1.00"
    score = dataset.score_answer(user_answer, item)
    assert score == 1.0, f"Expected score 1.0, got {score}"

    # Case 3: Negative base, valid exponent
    item = dataset[0].copy()
    item["answer"] = str(Decimal("-2.00") ** 3)  # -8.0
    user_answer = "-8.00"
    score = dataset.score_answer(user_answer, item)
    assert score == 1.0, f"Expected score 1.0, got {score}"

    # Case 4: Very small number with exponent notation
    item = dataset[0].copy()
    item["answer"] = str(Decimal("1e-6"))  # 1e-6
    user_answer = "1.00e-6"
    score = dataset.score_answer(user_answer, item)
    assert score == 1.0, f"Expected score 1.0, got {score}"

    # Case 5: Incorrect answer should yield low score
    item = dataset[0].copy()
    item["answer"] = "1000.0"
    user_answer = "999.0"
    score = dataset.score_answer(user_answer, item)
    assert score == 0.01, f"Expected low score 0.01, got {score}"
