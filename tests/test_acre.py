from random import Random

import pytest

from reasoning_gym.induction.acre.acre import ACREDataset, ACREDatasetConfig


def test_acre_config_validation():
    """Test that config validation works"""
    config = ACREDatasetConfig(size=-1)
    with pytest.raises(AssertionError):
        config.validate()


def test_acre_dataset_deterministic():
    """Test that dataset generates same items with same seed"""
    config = ACREDatasetConfig(seed=42, size=10)
    dataset1 = ACREDataset(config)
    dataset2 = ACREDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_acre_items():
    """Test basic properties of generated items"""
    config = ACREDatasetConfig(size=50, seed=42)
    dataset = ACREDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert isinstance(item["question"], str)
        assert isinstance(item["answer"], str)


def test_acre_iteration():
    """Test that iteration respects dataset size"""
    config = ACREDatasetConfig(size=10, seed=42)  # Small size for testing
    dataset = ACREDataset(config)

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


def test_acre_questions_generator():
    """Test question generator loading and access"""
    config = ACREDatasetConfig(size=10, seed=42)
    dataset = ACREDataset(config)

    # Test lazy loading
    assert dataset._questions is None
    _ = dataset.questions  # Access to trigger loading
    assert dataset._questions is not None

    # Test properties of questions
    assert isinstance(dataset.questions, list)
    assert len(dataset.questions) > 0
