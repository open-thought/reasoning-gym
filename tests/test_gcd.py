from functools import reduce
from math import gcd

import pytest

from reasoning_gym.arithmetic import GCDConfig, GCDDataset


def test_gcd_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = GCDConfig(min_numbers=1)  # Should be >= 2
        config.validate()

    with pytest.raises(AssertionError):
        config = GCDConfig(min_numbers=3, max_numbers=2)  # max should be >= min
        config.validate()

    with pytest.raises(AssertionError):
        config = GCDConfig(min_value=0)  # Should be positive
        config.validate()

    with pytest.raises(AssertionError):
        config = GCDConfig(min_value=100, max_value=50)  # max should be > min
        config.validate()


def test_gcd_deterministic():
    """Test that dataset generates same items with same seed"""
    config = GCDConfig(seed=42, size=10)
    dataset1 = GCDDataset(config)
    dataset2 = GCDDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_gcd_items():
    """Test basic properties of generated items"""
    config = GCDConfig(min_numbers=2, max_numbers=4, min_value=1, max_value=100, size=50, seed=42)
    dataset = GCDDataset(config)

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

        # Verify the GCD calculation is correct
        result = metadata["result"]
        assert str(result) == item["answer"]
        assert result == reduce(gcd, numbers)


def test_gcd_number_ranges():
    """Test that generated numbers respect value constraints"""
    config = GCDConfig(min_numbers=2, max_numbers=2, min_value=50, max_value=100, size=20, seed=42)
    dataset = GCDDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        numbers = item["metadata"]["numbers"]
        assert all(50 <= n <= 100 for n in numbers)


def test_gcd_iteration():
    """Test that iteration works correctly"""
    config = GCDConfig(size=5, seed=42)
    dataset = GCDDataset(config)

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


def test_gcd_special_cases():
    """Test some special GCD cases"""
    config = GCDConfig(min_numbers=2, max_numbers=2, min_value=1, max_value=100, size=100, seed=42)
    dataset = GCDDataset(config)

    # Track if we see some interesting GCD cases
    seen_gcd_1 = False  # Coprime numbers
    seen_large_gcd = False  # GCD > 1

    for i in range(len(dataset)):
        item = dataset[i]
        result = int(item["answer"])
        if result == 1:
            seen_gcd_1 = True
        if result > 1:
            seen_large_gcd = True

    # With enough samples, we should see both coprime and non-coprime numbers
    assert seen_gcd_1, "Expected to see some coprime numbers (GCD=1)"
    assert seen_large_gcd, "Expected to see some non-coprime numbers (GCD>1)"
