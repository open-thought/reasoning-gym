"""Tests for Number Format questions generation"""

import pytest

from reasoning_gym.arithmetic.number_format import NumberFormatConfig, NumberFormatCurriculum, NumberFormatDataset


def test_number_format_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = NumberFormatConfig(min_num_candidates=0)  # Zero not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = NumberFormatConfig(min_num_candidates=1)  # One not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = NumberFormatConfig(max_num_candidates=5, min_num_candidates=6)  # min > max
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
    assert dataset.score_answer(model_answer, entry) == 0.0

    # Answer is null
    model_answer = None
    assert dataset.score_answer(model_answer, entry) == 0.0

    # Answer is unparsable
    model_answer = "test"
    assert dataset.score_answer(model_answer, entry) == 0.0


def test_number_format_curriculum():
    curriculum = NumberFormatCurriculum()

    base_value = {"size": 150, "seed": 1}

    base_cfg: NumberFormatConfig = curriculum.generate_configuration(base_value)
    assert base_cfg.seed == 1
    assert base_cfg.size == 150
    assert base_cfg.min_num_candidates == 5 and base_cfg.max_num_candidates == 25
    assert base_cfg.min_n == 1000 and base_cfg.max_n == 100_000
    assert base_cfg.max_delta == 1e1

    # test incrementing attribute levels
    curriculum.increment_attr_level("num_candidates")
    curriculum.increment_attr_level("n")
    curriculum.increment_attr_level("max_delta")
    increased_cfg = curriculum.generate_configuration(base_value)
    assert increased_cfg.min_num_candidates == 5 and increased_cfg.max_num_candidates == 100
    assert increased_cfg.min_n == 1000 and increased_cfg.max_n == 1_000_000
    assert increased_cfg.max_delta == 1e0

    # test decrementing attribute level
    curriculum.decrement_attr_level("num_candidates")
    partially_decreased_cfg = curriculum.generate_configuration(base_value)
    assert partially_decreased_cfg.min_num_candidates == 5 and partially_decreased_cfg.max_num_candidates == 25
    assert partially_decreased_cfg.min_n == 1000 and partially_decreased_cfg.max_n == 1_000_000
    assert partially_decreased_cfg.max_delta == 1e0
