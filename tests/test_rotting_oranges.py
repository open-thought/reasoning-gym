"""Tests for Binary Matrix questions generation"""

import pytest

from reasoning_gym.algorithmic.rotten_oranges import RottenOrangesConfig, RottenOrangesDataset


def test_rotten_oranges_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = RottenOrangesConfig(max_n=-1)  # Negative not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = RottenOrangesConfig(max_n=0)  # Zero not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = RottenOrangesConfig(min_n=-1)  # Negative not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = RottenOrangesConfig(min_n=0)  # Zero not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = RottenOrangesConfig(p_oranges=0)  # <= 0 not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = RottenOrangesConfig(p_oranges=1.01)  # > 1 not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = RottenOrangesConfig(p_rotten=0)  # <= 0 not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = RottenOrangesConfig(p_rotten=1.01)  # > 1 not allowed
        config.validate()


def test_rotten_oranges_dataset_deterministic():
    """Test that dataset generates same items with same seed"""
    config = RottenOrangesConfig(seed=42, size=10)
    dataset1 = RottenOrangesDataset(config)
    dataset2 = RottenOrangesDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_rotten_oranges_dataset_items():
    """Test basic properties of generated items"""
    config = RottenOrangesConfig(min_n=10, max_n=15, size=10, seed=42)
    dataset = RottenOrangesDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        # Check item structure
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata
        assert "matrix" in item["metadata"]
        assert "solution" in item["metadata"]

        matrix = item["metadata"]["matrix"]

        # Verify dimensions
        assert config.min_n <= len(matrix) <= config.max_n
        assert all(config.min_n <= len(row) <= config.max_n for row in matrix)
        for r in range(len(matrix)):
            for c in range(len(matrix[0])):
                assert matrix[r][c] in [0, 1, 2]


def test_rotten_oranges_dataset_iteration():
    """Test that iteration respects dataset size"""
    config = RottenOrangesConfig(size=5, seed=42)
    dataset = RottenOrangesDataset(config)

    items = list(dataset)
    assert len(items) == config.size

    # Test multiple iterations yield same items
    assert items == list(dataset)


def test_rotten_oranges_answer():
    """Test the _get_distances method"""
    config = RottenOrangesConfig(seed=42)
    dataset = RottenOrangesDataset(config)

    # All oranges are rotten
    matrix = [
        [2, 2, 2],
        [2, 2, 2],
        [2, 2, 2],
    ]
    assert dataset._get_answer(matrix) == 0

    # All oranges are healthy
    matrix = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ]
    assert dataset._get_answer(matrix) == -1

    # 1 shot example
    matrix = [
        [2, 1, 1],
        [1, 1, 0],
        [0, 1, 1],
    ]
    assert dataset._get_answer(matrix) == 4
