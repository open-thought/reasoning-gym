"""Tests for Rotate Matrix questions generation"""

import pytest

from reasoning_gym.algorithmic.rotate_matrix import RotateMatrixConfig, RotateMatrixDataset


def test_rotate_matrix_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = RotateMatrixConfig(max_n=-1)  # Negative not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = RotateMatrixConfig(max_n=0)  # Zero not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = RotateMatrixConfig(max_rotations=-1)  # Negative not allowed
        config.validate()


def test_rotate_matrix_dataset_deterministic():
    """Test that dataset generates same items with same seed"""
    config = RotateMatrixConfig(seed=42, size=10)
    dataset1 = RotateMatrixDataset(config)
    dataset2 = RotateMatrixDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_rotate_matrix_dataset_items():
    """Test basic properties of generated items"""
    config = RotateMatrixConfig(max_n=7, size=10, seed=42)
    dataset = RotateMatrixDataset(config)

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
        num_rotations = item["metadata"]["num_rotations"]

        # Verify matrix dimensions
        assert len(matrix) <= config.max_n
        assert all(len(row) <= config.max_n for row in matrix)
        assert len(solution) <= config.max_n
        assert all(len(row) <= config.max_n for row in solution)
        assert set(e for row in matrix for e in row) == set(e for row in solution for e in row)
        assert num_rotations <= config.max_rotations


def test_rotate_matrix_dataset_iteration():
    """Test that iteration respects dataset size"""
    config = RotateMatrixConfig(size=5, seed=42)
    dataset = RotateMatrixDataset(config)

    items = list(dataset)
    assert len(items) == config.size

    # Test multiple iterations yield same items
    assert items == list(dataset)


def test_rotate_matrix_answer():
    """Test the _get_rotated method"""
    config = RotateMatrixConfig(seed=42)
    dataset = RotateMatrixDataset(config)

    # n = 1, num_rotations = 1
    matrix = [[8]]
    expected = [[8]]
    assert dataset._get_rotated(matrix, num_rotations=1) == expected

    # n = 3, num_rotations = 0 (no rotation)
    matrix = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
    ]
    expected = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
    ]
    assert dataset._get_rotated(matrix, num_rotations=0) == expected

    # n = 3, num_rotations = 1 (90 degrees clockwise)
    matrix = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
    ]
    expected = [
        [6, 3, 0],
        [7, 4, 1],
        [8, 5, 2],
    ]
    assert dataset._get_rotated(matrix, num_rotations=1) == expected

    # n = 3, num_rotations = 2 (180 degrees clockwise)
    matrix = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
    ]
    expected = [
        [8, 7, 6],
        [5, 4, 3],
        [2, 1, 0],
    ]
    assert dataset._get_rotated(matrix, num_rotations=2) == expected

    # n = 3, num_rotations = 3 (270 degrees clockwise)
    matrix = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
    ]
    expected = [[2, 5, 8], [1, 4, 7], [0, 3, 6]]
    assert dataset._get_rotated(matrix, num_rotations=3) == expected

    # n = 4, num_rotations = 4 (360 degrees clockwise == 0 degrees)
    matrix = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
    ]
    expected = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
    ]
    assert dataset._get_rotated(matrix, num_rotations=4) == expected
