"""Tests for Number Format questions generation"""

import pytest

from reasoning_gym.arithmetic.number_format import NumberFormatConfig, NumberFormatDataset


def test_number_format_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = NumberFormatConfig(max_num_candidates=0)  # Zero not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = NumberFormatConfig(max_num_candidates=1)  # One not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = NumberFormatConfig(min_n=-1)  # Negative not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = NumberFormatConfig(min_n=0)  # Zero not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = NumberFormatConfig(min_n=10, max_n=5)  # min > max
        config.validate()

    with pytest.raises(AssertionError):
        config = NumberFormatConfig(max_delta=-1)  # Negative not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = NumberFormatConfig(max_delta=0)  # Zero not allowed
        config.validate()


def test_number_format_dataset_deterministic():
    """Test that dataset generates same items with same seed"""
    config = NumberFormatConfig(seed=42, size=10)
    dataset1 = NumberFormatDataset(config)
    dataset2 = NumberFormatDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_number_format_dataset_items():
    """Test basic properties of generated items"""
    config = NumberFormatConfig(min_n=1_000, max_n=10_000, max_delta=1, size=10, seed=42)
    dataset = NumberFormatDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        # Check item structure
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata
        assert "candidates" in item["metadata"]
        assert "formatted_candidates" in item["metadata"]
        assert "size" in item["metadata"]
        assert "solution" in item["metadata"]

        candidates = item["metadata"]["candidates"]
        formatted_candidates = item["metadata"]["formatted_candidates"]
        size = item["metadata"]["size"]
        solution = item["metadata"]["solution"]

        # Verify values
        assert len(candidates) >= 2
        assert all(999 <= c <= 10_001 for c in candidates)  # boundaries +- delta
        assert len(candidates) == len(formatted_candidates)
        assert size in ["largest", "smallest"]
        assert solution in candidates


def test_number_format_dataset_iteration():
    """Test that iteration respects dataset size"""
    config = NumberFormatConfig(size=5, seed=42)
    dataset = NumberFormatDataset(config)

    items = list(dataset)
    assert len(items) == config.size

    # Test multiple iterations yield same items
    assert items == list(dataset)


def test_number_format_answer():
    """Verify the solution scoring"""
    config = NumberFormatConfig(size=5, seed=42)
    dataset = NumberFormatDataset(config)

    entry = {"metadata": {"solution": 54245.32}}

    # Correct answer (plain)
    model_answer = "54245.32"
    assert dataset.score_answer(model_answer, entry) == 1.0

    # Correct answer (English)
    model_answer = "54,245.32"
    assert dataset.score_answer(model_answer, entry) == 1.0

    # Correct answer (scientific)
    assert dataset.score_answer("5.424532e+04", entry) == 1.0

    # Incorrect answer (diff larger than 1e-2)
    model_answer = "54245.9"
    assert dataset.score_answer(model_answer, entry) == 0.01

    # Answer is null
    model_answer = None
    assert dataset.score_answer(model_answer, entry) == 0.0

    # Answer is unparsable
    model_answer = "test"
    assert dataset.score_answer(model_answer, entry) == 0.0
