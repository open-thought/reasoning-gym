from math import gcd

import pytest

from reasoning_gym.arithmetic import FractionSimplificationConfig, FractionSimplificationDataset


def test_fraction_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = FractionSimplificationConfig(min_value=0)  # Should be positive
        config.validate()

    with pytest.raises(AssertionError):
        config = FractionSimplificationConfig(min_value=100, max_value=50)  # max should be > min
        config.validate()

    with pytest.raises(AssertionError):
        config = FractionSimplificationConfig(min_factor=0)  # Should be >= 1
        config.validate()

    with pytest.raises(AssertionError):
        config = FractionSimplificationConfig(min_factor=5, max_factor=3)  # max should be >= min
        config.validate()


def test_fraction_deterministic():
    """Test that dataset generates same items with same seed"""
    config = FractionSimplificationConfig(seed=42, size=10)
    dataset1 = FractionSimplificationDataset(config)
    dataset2 = FractionSimplificationDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_fraction_items():
    """Test basic properties of generated items"""
    config = FractionSimplificationConfig(min_value=1, max_value=20, min_factor=2, max_factor=5, size=50, seed=42)
    dataset = FractionSimplificationDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Verify the metadata contains all expected fields
        metadata = item["metadata"]
        assert "numerator" in metadata
        assert "denominator" in metadata
        assert "simplified_numerator" in metadata
        assert "simplified_denominator" in metadata
        assert "reduction_factor" in metadata

        # Verify the numbers are within configured range
        assert config.min_value <= metadata["simplified_numerator"] <= config.max_value
        assert config.min_value <= metadata["simplified_denominator"] <= config.max_value

        # Verify the reduction is correct
        num = metadata["numerator"]
        den = metadata["denominator"]
        simple_num = metadata["simplified_numerator"]
        simple_den = metadata["simplified_denominator"]
        factor = metadata["reduction_factor"]

        assert num == simple_num * factor
        assert den == simple_den * factor

        # Verify the simplified fraction is actually in lowest terms
        assert gcd(simple_num, simple_den) == 1


def test_fraction_ranges():
    """Test that generated numbers respect value constraints"""
    config = FractionSimplificationConfig(min_value=5, max_value=15, min_factor=3, max_factor=4, size=20, seed=42)
    dataset = FractionSimplificationDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        metadata = item["metadata"]
        factor = metadata["reduction_factor"]

        # Check factor is within bounds
        assert 3 <= factor <= 4

        # Check simplified values are within bounds
        assert 5 <= metadata["simplified_numerator"] <= 15
        assert 5 <= metadata["simplified_denominator"] <= 15


def test_fraction_iteration():
    """Test that iteration works correctly"""
    config = FractionSimplificationConfig(size=5, seed=42)
    dataset = FractionSimplificationDataset(config)

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


def test_fraction_numerator_smaller():
    """Test that numerators are always smaller than denominators"""
    config = FractionSimplificationConfig(min_value=1, max_value=100, min_factor=2, max_factor=5, size=50, seed=42)
    dataset = FractionSimplificationDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        metadata = item["metadata"]

        # Check original fraction
        assert (
            metadata["numerator"] <= metadata["denominator"]
        ), f"Original numerator {metadata['numerator']} should be <= denominator {metadata['denominator']}"

        # Check simplified fraction
        assert (
            metadata["simplified_numerator"] <= metadata["simplified_denominator"]
        ), f"Simplified numerator {metadata['simplified_numerator']} should be <= denominator {metadata['simplified_denominator']}"
