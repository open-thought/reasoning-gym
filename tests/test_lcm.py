from functools import reduce
from math import lcm

import pytest

from reasoning_gym.arithmetic import LCMConfig, LCMDataset


def test_lcm_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = LCMConfig(min_numbers=1)  # Should be >= 2
        config.validate()

    with pytest.raises(AssertionError):
        config = LCMConfig(min_numbers=3, max_numbers=2)  # max should be >= min
        config.validate()

    with pytest.raises(AssertionError):
        config = LCMConfig(min_value=0)  # Should be positive
        config.validate()

    with pytest.raises(AssertionError):
        config = LCMConfig(min_value=100, max_value=50)  # max should be > min
        config.validate()


def test_lcm_deterministic():
    """Test that dataset generates same items with same seed"""
    config = LCMConfig(seed=42, size=10)
    dataset1 = LCMDataset(config)
    dataset2 = LCMDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_lcm_items():
    """Test basic properties of generated items"""
    config = LCMConfig(
        min_numbers=2, max_numbers=4, min_value=1, max_value=20, size=50, seed=42  # Keep small for testing
    )
    dataset = LCMDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Verify the numbers and result are in metadata
        metadata = item["metadata"]
        assert "numbers" in metadata
        assert "result" in metadata

        # Verify the numbers are within configured range
        numbers = metadata["numbers"]
        assert all(config.min_value <= n <= config.max_value for n in numbers)
        assert config.min_numbers <= len(numbers) <= config.max_numbers

        # Verify the LCM calculation is correct
        result = metadata["result"]
        assert str(result) == item["answer"]
        assert result == reduce(lcm, numbers)


def test_lcm_number_ranges():
    """Test that generated numbers respect value constraints"""
    config = LCMConfig(min_numbers=2, max_numbers=2, min_value=5, max_value=15, size=20, seed=42)
    dataset = LCMDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        numbers = item["metadata"]["numbers"]
        assert all(5 <= n <= 15 for n in numbers)


def test_lcm_iteration():
    """Test that iteration works correctly"""
    config = LCMConfig(size=5, seed=42)
    dataset = LCMDataset(config)

    # Test manual iteration
    items = []
    for item in dataset:
        items.append(item)
    assert len(items) == config.size

    # Test list conversion
    items = list(dataset)
    assert len(items) == config.size

    # Test multiple iterations yield same results
    first_items = list(dataset)
    second_items = list(dataset)
    assert first_items == second_items


def test_lcm_special_cases():
    """Test some special LCM cases"""
    config = LCMConfig(min_numbers=2, max_numbers=2, min_value=1, max_value=20, size=100, seed=42)
    dataset = LCMDataset(config)

    # Track if we see some interesting LCM cases
    seen_equal_to_product = False  # When numbers are coprime
    seen_less_than_product = False  # When numbers share factors

    for i in range(len(dataset)):
        item = dataset[i]
        numbers = item["metadata"]["numbers"]
        result = int(item["answer"])
        product = reduce(lambda x, y: x * y, numbers)

        if result == product:
            seen_equal_to_product = True
        if result < product:
            seen_less_than_product = True

    # With enough samples, we should see both cases
    assert seen_equal_to_product, "Expected to see some coprime numbers (LCM = product)"
    assert seen_less_than_product, "Expected to see some numbers with common factors (LCM < product)"
