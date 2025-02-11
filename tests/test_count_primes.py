"""Tests for Count Primes questions generation"""

import pytest

from reasoning_gym.algorithmic.count_primes import CountPrimesConfig, CountPrimesDataset


def test_count_primes_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = CountPrimesConfig(max_n=-1)  # Negative not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = CountPrimesConfig(max_n=0)  # Zero not allowed
        config.validate()


def test_count_primes_dataset_deterministic():
    """Test that dataset generates same items with same seed"""
    config = CountPrimesConfig(seed=42, size=10)
    dataset1 = CountPrimesDataset(config)
    dataset2 = CountPrimesDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_count_primes_dataset_items():
    """Test basic properties of generated items"""
    config = CountPrimesConfig(max_n=10, size=10, seed=42)
    dataset = CountPrimesDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        # Check item structure
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata
        assert "start" in item["metadata"]
        assert "end" in item["metadata"]
        assert "primes" in item["metadata"]
        assert "solution" in item["metadata"]

        start = item["metadata"]["start"]
        end = item["metadata"]["end"]
        primes = item["metadata"]["primes"]

        assert start <= end
        assert len(primes) <= end - start + 1


def test_count_primes_dataset_iteration():
    """Test that iteration respects dataset size"""
    config = CountPrimesConfig(size=5, seed=42)
    dataset = CountPrimesDataset(config)

    items = list(dataset)
    assert len(items) == config.size

    # Test multiple iterations yield same items
    assert items == list(dataset)


def test_count_primes_answer():
    """Test the _get_primes method"""
    config = CountPrimesConfig(seed=42)
    dataset = CountPrimesDataset(config)

    # Base cases
    assert dataset._get_primes(n=0) == []
    assert dataset._get_primes(n=1) == []
    assert dataset._get_primes(n=2) == [False, False]

    # Test primes up to 10
    primes = dataset._get_primes(n=11)
    assert primes[2] == True
    assert primes[3] == True
    assert primes[4] == False
    assert primes[5] == True
    assert primes[6] == False
    assert primes[7] == True
    assert primes[8] == False
    assert primes[9] == False
    assert primes[10] == False
