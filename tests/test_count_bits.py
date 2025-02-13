"""Tests for Count bits questions generation"""

import pytest

from reasoning_gym.arithmetic.count_bits import CountBitsConfig, CountBitsDataset


def test_count_bits_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = CountBitsConfig(max_n=-1)  # Negative not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = CountBitsConfig(max_n=0)  # Zero not allowed
        config.validate()


def test_count_bits_dataset_deterministic():
    """Test that dataset generates same items with same seed"""
    config = CountBitsConfig(seed=42, size=10)
    dataset1 = CountBitsDataset(config)
    dataset2 = CountBitsDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_count_bits_dataset_items():
    """Test basic properties of generated items"""
    config = CountBitsConfig(max_n=10, size=10, seed=42)
    dataset = CountBitsDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        # Check item structure
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata
        assert "number" in item["metadata"]
        assert "solution" in item["metadata"]
        assert "binary" in item["metadata"]

        number = item["metadata"]["number"]
        solution = item["metadata"]["solution"]
        binary = item["metadata"]["binary"]

        # Verify values
        assert number <= config.max_n
        assert solution >= 0
        assert set(binary) <= {"0", "1"}


def test_count_bits_dataset_iteration():
    """Test that iteration respects dataset size"""
    config = CountBitsConfig(size=5, seed=42)
    dataset = CountBitsDataset(config)

    items = list(dataset)
    assert len(items) == config.size

    # Test multiple iterations yield same items
    assert items == list(dataset)


def test_count_bits_answer():
    """Verify the number of 1 bits in the binary representation of a number"""
    config = CountBitsConfig(size=5, seed=42)
    dataset = CountBitsDataset(config)

    for item in dataset:
        number = item["metadata"]["number"]
        solution = item["metadata"]["solution"]

        # Count number of 1 bits in the number by shifting
        count = 0
        while number:
            count += number & 1
            number >>= 1
        assert solution == count
