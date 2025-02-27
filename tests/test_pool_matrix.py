"""Tests for Pool Matrix questions generation"""

import numpy as np
import pytest

from reasoning_gym.algorithmic.pool_matrix import PoolMatrixConfig, PoolMatrixDataset


def test_pool_matrix_config_validation():
    """Test that invalid configs raise appropriate errors"""

    for field in ["min_rows", "min_cols", "max_rows", "max_cols"]:
        with pytest.raises(AssertionError):
            config = PoolMatrixConfig(**{field: -1})  # Negative not allowed
            config.validate()

        with pytest.raises(AssertionError):
            config = PoolMatrixConfig(**{field: 0})  # Zero not allowed
            config.validate()

        with pytest.raises(AssertionError):
            config = PoolMatrixConfig(**{field: 1})  # One not allowed
            config.validate()

    with pytest.raises(AssertionError):
        config = PoolMatrixConfig(max_pool_size=-1)  # Negative not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = PoolMatrixConfig(max_pool_size=0)  # Zero not allowed
        config.validate()


def test_pool_matrix_dataset_deterministic():
    """Test that dataset generates same items with same seed"""
    config = PoolMatrixConfig(seed=42, size=10)
    dataset1 = PoolMatrixDataset(config)
    dataset2 = PoolMatrixDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_pool_matrix_dataset_items():
    """Test basic properties of generated items"""
    config = PoolMatrixConfig(max_rows=10, max_cols=10, max_pool_size=3, size=10, seed=42)
    dataset = PoolMatrixDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        # Check item structure
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata
        assert "matrix" in item["metadata"]
        assert "pool_type" in item["metadata"]
        assert "pool_size" in item["metadata"]
        assert "solution" in item["metadata"]

        matrix = item["metadata"]["matrix"]
        pool_type = item["metadata"]["pool_type"]
        pool_size = item["metadata"]["pool_size"]
        solution = item["metadata"]["solution"]

        # Verify dimensions
        assert len(matrix) <= config.max_rows
        assert all(len(row) <= config.max_cols for row in matrix)
        assert len(solution) <= len(matrix)
        assert len(solution[0]) <= len(matrix[0])
        assert pool_size <= config.max_pool_size
        assert pool_type in ["average", "max"]


def test_pool_matrix_dataset_iteration():
    """Test that iteration respects dataset size"""
    config = PoolMatrixConfig(size=5, seed=42)
    dataset = PoolMatrixDataset(config)

    items = list(dataset)
    assert len(items) == config.size

    # Test multiple iterations yield same items
    assert items == list(dataset)


def test_pool_matrix_answer():
    """Test the pooling methods"""
    config = PoolMatrixConfig(seed=42)
    dataset = PoolMatrixDataset(config)

    # 1. Max pooling
    matrix = np.array([[1, 2], [3, 4]])
    assert np.allclose(dataset._max_pool(matrix, 2), np.array([[4]]))

    matrix = np.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
        ]
    )
    assert np.allclose(dataset._max_pool(matrix, 2), np.array([[6, 8], [10, 12]]))

    matrix = np.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ]
    )
    assert np.allclose(dataset._max_pool(matrix, 2), np.array([[6, 8], [14, 16]]))

    # 2. Average pooling
    matrix = np.array([[1, 2], [3, 4]])
    assert np.allclose(dataset._average_pool(matrix, 2), np.array([[2.5]]))

    matrix = np.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
        ]
    )
    assert np.allclose(dataset._average_pool(matrix, 2), np.array([[3.5, 5.5], [9.5, 11.5]]))

    matrix = np.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ]
    )
    assert np.allclose(dataset._average_pool(matrix, 2), np.array([[3.5, 5.5], [11.5, 13.5]]))


def test_pool_matrix_score_answer():
    config = PoolMatrixConfig(seed=42, size=100)
    dataset = PoolMatrixDataset(config)
    for entry in dataset:
        assert dataset.score_answer(entry["answer"], entry=entry) == 1
        assert 0.0 < dataset.score_answer("1 2.0\n3.0 4", entry=entry) <= 0.1
        assert dataset.score_answer("one two three", entry=entry) == 0.0
        assert dataset.score_answer("", entry=entry) == 0.0
        assert dataset.score_answer(None, entry=entry) == 0.0


def test_pool_matrix_int_answer():
    config = PoolMatrixConfig(seed=42, size=10)
    dataset = PoolMatrixDataset(config)
    for entry in dataset:
        matrix = np.loadtxt(entry["answer"].splitlines())
        is_integer = np.equal(np.mod(matrix, 1), 0)
        if is_integer.all():
            matrix = matrix.astype(np.int32)
            if matrix.ndim == 0:
                matrix = matrix.reshape(1, 1)
            int_answer = "\n".join(" ".join(str(x) for x in row) for row in matrix)
            assert dataset.score_answer(answer=int_answer, entry=entry) == 1.0
