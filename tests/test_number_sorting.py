"""Tests for number sorting task generation"""

import pytest

from reasoning_gym.algorithmic.number_sorting import NumberSortingConfig, NumberSortingDataset


def test_number_sorting_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = NumberSortingConfig(min_numbers=0)
        config.validate()

    with pytest.raises(AssertionError):
        config = NumberSortingConfig(min_numbers=10, max_numbers=5)
        config.validate()

    with pytest.raises(AssertionError):
        config = NumberSortingConfig(min_decimals=-1)
        config.validate()

    with pytest.raises(AssertionError):
        config = NumberSortingConfig(min_value=100, max_value=0)
        config.validate()


def test_number_sorting_dataset_deterministic():
    """Test that dataset generates same items with same seed"""
    config = NumberSortingConfig(seed=42, size=10)
    dataset1 = NumberSortingDataset(config)
    dataset2 = NumberSortingDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_number_sorting_dataset_items():
    """Test basic properties of generated items"""
    config = NumberSortingConfig(
        min_numbers=3, max_numbers=6, min_decimals=1, max_decimals=3, min_value=-10.0, max_value=10.0, size=10, seed=42
    )
    dataset = NumberSortingDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        # Check item structure
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata
        assert "original_numbers" in item["metadata"]
        assert "direction" in item["metadata"]
        assert "sorted_numbers" in item["metadata"]

        # Verify number count constraints
        numbers = item["metadata"]["original_numbers"]
        assert len(numbers) >= config.min_numbers
        assert len(numbers) <= config.max_numbers

        # Verify decimal places
        for num in numbers:
            decimal_places = len(num.split(".")[-1]) if "." in num else 0
            assert decimal_places >= config.min_decimals
            assert decimal_places <= config.max_decimals

        # Verify value range
        for num in numbers:
            value = float(num)
            assert config.min_value <= value <= config.max_value

        # Verify sorting
        direction = item["metadata"]["direction"]
        sorted_numbers = [float(x) for x in eval(item["answer"])]
        if direction == "ascending":
            assert sorted_numbers == sorted(sorted_numbers)
        else:
            assert sorted_numbers == sorted(sorted_numbers, reverse=True)


def test_number_sorting_dataset_iteration():
    """Test that iteration respects dataset size"""
    config = NumberSortingConfig(size=5, seed=42)
    dataset = NumberSortingDataset(config)

    items = list(dataset)
    assert len(items) == config.size

    # Test multiple iterations yield same items
    assert items == list(dataset)
