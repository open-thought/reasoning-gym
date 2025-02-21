"""Tests for Binary Alternation questions generation"""

import pytest

from reasoning_gym.algorithmic.binary_alternation import BinaryAlternationConfig, BinaryAlternationDataset


def test_binary_alternation_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = BinaryAlternationConfig(max_n=-1)  # Negative not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = BinaryAlternationConfig(max_n=0)  # Zero not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = BinaryAlternationConfig(min_n=-1)  # Negative not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = BinaryAlternationConfig(min_n=0)  # Zero not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = BinaryAlternationConfig(p_solvable=-0.01)  # < 0 not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = BinaryAlternationConfig(p_solvable=1.01)  # > 0 not allowed
        config.validate()


def test_binary_alternation_dataset_deterministic():
    """Test that dataset generates same items with same seed"""
    config = BinaryAlternationConfig(seed=42, size=10)
    dataset1 = BinaryAlternationDataset(config)
    dataset2 = BinaryAlternationDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_binary_alternation_dataset_items():
    """Test basic properties of generated items"""
    config = BinaryAlternationConfig(max_n=10, size=10, seed=42)
    dataset = BinaryAlternationDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        # Check item structure
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata
        assert "string" in item["metadata"]
        assert "solution" in item["metadata"]
        assert "solvable" in item["metadata"]

        solution = item["metadata"]["solution"]
        string = item["metadata"]["string"]
        solvable = item["metadata"]["solvable"]

        # Verify values
        assert set(string) <= {"0", "1"}
        if solution == -1:
            assert not solvable
            assert abs(string.count("1") - string.count("0")) > 1
        else:
            assert solvable
            assert abs(string.count("1") - string.count("0")) <= 1


def test_binary_alternation_dataset_iteration():
    """Test that iteration respects dataset size"""
    config = BinaryAlternationConfig(size=5, seed=42)
    dataset = BinaryAlternationDataset(config)

    items = list(dataset)
    assert len(items) == config.size

    # Test multiple iterations yield same items
    assert items == list(dataset)


def test_binary_alternation_answer():
    """Verify the number of 1 bits in the binary representation of a number"""
    config = BinaryAlternationConfig(size=5, seed=42)
    dataset = BinaryAlternationDataset(config)

    # Impossible
    string = "1110"
    assert dataset._get_answer(string) == -1

    # Already alternating
    string = "10101"
    assert dataset._get_answer(string) == 0

    # One shot example
    string = "111000"
    assert dataset._get_answer(string) == 1
