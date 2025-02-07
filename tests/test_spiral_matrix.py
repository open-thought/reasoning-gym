"""Tests for Spiral Matrix questions generation"""

import pytest

from reasoning_gym.algorithmic.spiral_matrix import SpiralMatrixConfig, SpiralMatrixDataset


def test_spiral_matrix_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = SpiralMatrixConfig(max_rows=-1)  # Negative not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = SpiralMatrixConfig(max_rows=0)  # Zero not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = SpiralMatrixConfig(max_cols=-1)  # Negative not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = SpiralMatrixConfig(max_cols=0)  # Zero not allowed
        config.validate()


def test_spiral_matrix_dataset_deterministic():
    """Test that dataset generates same items with same seed"""
    config = SpiralMatrixConfig(seed=42, size=10)
    dataset1 = SpiralMatrixDataset(config)
    dataset2 = SpiralMatrixDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_spiral_matrix_dataset_items():
    """Test basic properties of generated items"""
    config = SpiralMatrixConfig(max_rows=5, max_cols=5, size=10, seed=42)
    dataset = SpiralMatrixDataset(config)

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
        solution = item["metadata"]["solution"]

        # Verify list dimensions
        assert len(matrix) <= config.max_rows
        assert all(len(row) <= config.max_cols for row in matrix)
        assert sum(len(row) for row in matrix) == len(solution)
        assert len(list(set(solution))) == len(solution)


def test_spiral_matrix_dataset_iteration():
    """Test that iteration respects dataset size"""
    config = SpiralMatrixConfig(size=5, seed=42)
    dataset = SpiralMatrixDataset(config)

    items = list(dataset)
    assert len(items) == config.size

    # Test multiple iterations yield same items
    assert items == list(dataset)


def test_spiral_matrix_answer():
    """Test the _get_spiral method"""
    config = SpiralMatrixConfig(seed=42)
    dataset = SpiralMatrixDataset(config)

    # One element
    matrix = [[0]]
    assert dataset._get_spiral(matrix) == [0]

    # One row
    matrix = [[0, 1, 2]]
    assert dataset._get_spiral(matrix) == [0, 1, 2]

    # One column
    matrix = [[0], [1], [2]]
    assert dataset._get_spiral(matrix) == [0, 1, 2]

    # 2D grid
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert dataset._get_spiral(matrix) == [1, 2, 3, 6, 9, 8, 7, 4, 5]
