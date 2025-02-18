"""Tests for Isomorphic Strings questions generation"""

import json

import pytest

from reasoning_gym.algorithmic.isomorphic_strings import IsomorphicStringsConfig, IsomorphicStringsDataset


def test_isomorphic_strings_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = IsomorphicStringsConfig(max_string_length=-1)  # Negative not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = IsomorphicStringsConfig(max_string_length=0)  # Zero not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = IsomorphicStringsConfig(max_string_length=1)  # One not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = IsomorphicStringsConfig(p_solvable=-0.01)  # < 0 not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = IsomorphicStringsConfig(p_solvable=1.01)  # > 1 not allowed
        config.validate()


def test_isomorphic_strings_dataset_deterministic():
    """Test that dataset generates same items with same seed"""
    config = IsomorphicStringsConfig(seed=42, size=10)
    dataset1 = IsomorphicStringsDataset(config)
    dataset2 = IsomorphicStringsDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_isomorphic_strings_dataset_items():
    """Test basic properties of generated items"""
    config = IsomorphicStringsConfig(max_string_length=10, size=10, seed=42)
    dataset = IsomorphicStringsDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        # Check item structure
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata
        assert "words" in item["metadata"]
        assert "solution" in item["metadata"]
        assert "solvable" in item["metadata"]

        words = item["metadata"]["words"]
        solution = item["metadata"]["solution"]
        solvable = item["metadata"]["solvable"]

        # Verify list dimensions
        assert len(words) == 2
        assert solution in {True, False}
        assert solvable in {True, False}
        assert solution == solvable


def test_isomorphic_strings_dataset_iteration():
    """Test that iteration respects dataset size"""
    config = IsomorphicStringsConfig(size=5, seed=42)
    dataset = IsomorphicStringsDataset(config)

    items = list(dataset)
    assert len(items) == config.size

    # Test multiple iterations yield same items
    assert items == list(dataset)


def test_isomorphic_strings_answer():
    """Test the _check_isomorphic method"""
    config = IsomorphicStringsConfig(seed=42)
    dataset = IsomorphicStringsDataset(config)

    # General use case
    s, t = "foo", "bar"
    assert dataset._check_isomorphic(s, t) == False

    s, t = "foo", "baa"
    assert dataset._check_isomorphic(s, t) == True

    # Unequal lengths
    s, t = "foo", "bo"
    assert dataset._check_isomorphic(s, t) == False

    # Empty strings
    (
        s,
        t,
    ) = (
        "",
        "",
    )
    assert dataset._check_isomorphic(s, t) == True
