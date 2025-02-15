"""Tests for Binary Matrix questions generation"""

import pytest

from reasoning_gym.algorithmic.binary_matrix import BinaryMatrixConfig, BinaryMatrixDataset


def test_binary_matrix_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = BinaryMatrixConfig(max_n=-1)  # Negative not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = BinaryMatrixConfig(max_n=0)  # Zero not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = BinaryMatrixConfig(min_n=-1)  # Negative not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = BinaryMatrixConfig(min_n=0)  # Zero not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = BinaryMatrixConfig(p_zero=0)  # <= 0 not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = BinaryMatrixConfig(p_zero=1.01)  # > 1 not allowed
        config.validate()


def test_binary_matrix_dataset_deterministic():
    """Test that dataset generates same items with same seed"""
    config = BinaryMatrixConfig(seed=42, size=10)
    dataset1 = BinaryMatrixDataset(config)
    dataset2 = BinaryMatrixDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_binary_matrix_dataset_items():
    """Test basic properties of generated items"""
    config = BinaryMatrixConfig(max_n=5, size=10, seed=42)
    dataset = BinaryMatrixDataset(config)

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
        assert len(matrix) <= config.max_n
        assert all(len(row) <= config.max_n for row in matrix)
        assert all(len(row) <= config.max_n for row in solution)

        # Verify matrix values
        for r in range(len(matrix)):
            for c in range(len(matrix[r])):
                assert matrix[r][c] in {0, 1}
                assert solution[r][c] >= matrix[r][c]


def test_binary_matrix_dataset_iteration():
    """Test that iteration respects dataset size"""
    config = BinaryMatrixConfig(size=5, seed=42)
    dataset = BinaryMatrixDataset(config)

    items = list(dataset)
    assert len(items) == config.size

    # Test multiple iterations yield same items
    assert items == list(dataset)


def test_binary_matrix_answer():
    """Test the _get_distances method"""
    config = BinaryMatrixConfig(seed=42)
    dataset = BinaryMatrixDataset(config)

    # 1x1 matrix
    matrix = [[0]]
    assert dataset._get_distances(matrix) == [[0]]

    # 2x2 matrix
    matrix = [[0, 1], [1, 1]]
    assert dataset._get_distances(matrix) == [[0, 1], [1, 2]]

    # 3x3 matrix
    matrix = [[0, 0, 0], [0, 1, 0], [1, 1, 1]]
    assert dataset._get_distances(matrix) == [[0, 0, 0], [0, 1, 0], [1, 2, 1]]

    # Empty matrix
    matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    assert dataset._get_distances(matrix) == [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    # String representation of answer
    answer = "0 0 0\n0 1 0\n1 2 1"
    entry = {"answer": "0 0 0\n0 1 0\n1 2 1"}
    assert dataset.score_answer(answer, entry) == 1.0

    # Answer is a python list (partially correct answer)
    answer = "[[0, 0, 0], [0, 1, 0], [1, 2, 1]]"
    entry = {"answer": "0 0 0\n0 1 0\n1 2 1"}
    assert dataset.score_answer(answer, entry) == 0.5

    # Answer is null
    answer = None
    entry = {"answer": "0 0 0\n0 1 0\n1 2 1"}
    assert dataset.score_answer(answer, entry) == 0.0
