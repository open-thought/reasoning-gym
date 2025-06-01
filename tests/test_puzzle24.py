import re

import pytest
from sympy.parsing.sympy_parser import parse_expr

from reasoning_gym.games.puzzle24 import Puzzle24Config, Puzzle24Curriculum, Puzzle24Dataset


def test_puzzle24_config_validation():
    """Test that invalid configs raise appropriate errors"""
    # Min value greater than max value
    with pytest.raises(AssertionError):
        config = Puzzle24Config(min_value=10, max_value=5)
        config.validate()

    # Invalid operators
    with pytest.raises(AssertionError):
        config = Puzzle24Config(operators=())  # Empty operators
        config.validate()

    # Negative min value
    with pytest.raises(AssertionError):
        config = Puzzle24Config(min_value=-1)
        config.validate()


def test_puzzle24_deterministic():
    """Test that dataset generates same items with same seed"""
    config = Puzzle24Config(seed=42, size=10)
    dataset1 = Puzzle24Dataset(config)
    dataset2 = Puzzle24Dataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_puzzle24_basic_properties():
    """Test basic properties of generated items"""
    config = Puzzle24Config(seed=42, size=10)
    dataset = Puzzle24Dataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata contains required fields
        assert "numbers" in item["metadata"]

        # Check question format
        assert "Make 24 using" in item["question"]
        assert "Final answer format instructions" in item["question"]

        # Verify the numbers in question match metadata
        numbers = item["metadata"]["numbers"]
        num_str = ", ".join(map(str, numbers))
        assert num_str in item["question"]


def test_puzzle24_score_answer_correct():
    """Test the score_answer method for correct answers"""
    config = Puzzle24Config(seed=42)
    dataset = Puzzle24Dataset(config)
    for item in dataset:
        answer = item["answer"]
        print(item)
        assert dataset.score_answer(answer, item) == 1.0


def test_puzzle24_score_answer_individual():
    """Test the score_answer method"""
    config = Puzzle24Config(seed=42)
    dataset = Puzzle24Dataset(config)

    # Create a test entry
    entry = {"metadata": {"numbers": [4, 5, 7, 8], "expression": parse_expr("x0 + x1 + x2 + x3")}}

    # Test correct answer (evaluates to 24)
    assert dataset.score_answer("4 + 5 + 7 + 8", entry) == 1.0

    # Test incorrect answers
    assert dataset.score_answer(None, entry) == 0.01  # None answer
    assert dataset.score_answer("", entry) == 0.01  # Empty answer
    assert dataset.score_answer("1+2+3", entry) == 0.01  # Wrong numbers
    assert dataset.score_answer("4*5*7*8", entry) == 0.01  # Doesn't equal 24
    assert dataset.score_answer("not a valid expression", entry) == 0.01  # Invalid expression


def test_puzzle24_iteration():
    """Test that iteration respects dataset size"""
    config = Puzzle24Config(size=5, seed=42)  # Small size for testing
    dataset = Puzzle24Dataset(config)

    # Test manual iteration
    items = []
    for item in dataset:
        items.append(item)
    assert len(items) == config.size, "Iterator should yield exactly size items"

    # Test list conversion
    items = list(dataset)
    assert len(items) == config.size, "Iterator should yield exactly size items"

    # Test multiple iterations
    first_items = list(dataset)
    second_items = list(dataset)
    assert first_items == second_items, "Multiple iterations should yield same items"


def test_puzzle24_curriculum():
    curriculum = Puzzle24Curriculum()

    base_value = {"size": 150, "seed": 1}

    base_cfg: Puzzle24Config = curriculum.generate_configuration(base_value)
    assert base_cfg.seed == 1
    assert base_cfg.size == 150
    assert base_cfg.min_value == 1 and base_cfg.max_value == 5

    # Test incrementing attribute levels
    curriculum.increment_attr_level("value")
    increased_cfg = curriculum.generate_configuration(base_value)
    assert increased_cfg.min_value == 1 and increased_cfg.max_value == 6

    # Test decrementing attribute levels
    curriculum.decrement_attr_level("value")
    partially_decreased_cfg = curriculum.generate_configuration(base_value)
    assert partially_decreased_cfg.min_value == 1 and partially_decreased_cfg.max_value == 5
