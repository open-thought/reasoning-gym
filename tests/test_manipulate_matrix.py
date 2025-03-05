"""Tests for Manipulate Matrix questions generation"""

import pytest

from reasoning_gym.algorithmic.manipulate_matrix import ManipulateMatrixConfig, ManipulateMatrixDataset


def test_manipulate_matrix_config_validation():
    """Test that invalid configs raise appropriate errors"""

    for field in ["min_transforms", "max_transforms"]:
        for num_transforms in [-1, 0]:  # Number of transforms should be positive
            with pytest.raises(AssertionError):
                config = ManipulateMatrixConfig(**{field: num_transforms})
                config.validate()

    invalid_dims = [-1, 0]  # Dimensions should be positive integers
    dim_fields = ["min_rows", "min_cols", "max_rows", "max_cols"]

    for field in dim_fields:
        for dim in invalid_dims:
            with pytest.raises(AssertionError):
                config = ManipulateMatrixConfig(**{field: dim})
                config.validate()


def test_manipulate_matrix_dataset_deterministic():
    """Test that dataset generates same items with same seed"""
    config = ManipulateMatrixConfig(seed=42, size=10)
    dataset1 = ManipulateMatrixDataset(config)
    dataset2 = ManipulateMatrixDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_manipulate_matrix_dataset_items():
    """Test basic properties of generated items"""
    config = ManipulateMatrixConfig(max_rows=7, max_cols=7, size=10, seed=42)
    dataset = ManipulateMatrixDataset(config)

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
        assert "operations" in item["metadata"]

        matrix = item["metadata"]["matrix"]
        solution = item["metadata"]["solution"]
        operations = item["metadata"]["operations"]

        # Verify matrix dimensions
        assert len(matrix) <= config.max_rows
        assert all(len(row) <= config.max_cols for row in matrix)
        assert len(solution) <= config.max_rows
        assert all(len(row) <= config.max_cols for row in solution)
        for op in operations:
            assert "transform" in op
            assert "instruction" in op


def test_manipulate_matrix_dataset_iteration():
    """Test that iteration respects dataset size"""
    config = ManipulateMatrixConfig(size=5, seed=42)
    dataset = ManipulateMatrixDataset(config)

    items = list(dataset)
    assert len(items) == config.size

    assert items == list(dataset)


def test_manipulate_matrix_transforms():
    """Test the _get_rotated method"""
    config = ManipulateMatrixConfig(seed=42)
    dataset = ManipulateMatrixDataset(config)
    matrix = [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25],
    ]

    # identity
    assert dataset._identity(matrix) == matrix

    # rot 90 degrees
    assert dataset._rot90(matrix) == [
        [21, 16, 11, 6, 1],
        [22, 17, 12, 7, 2],
        [23, 18, 13, 8, 3],
        [24, 19, 14, 9, 4],
        [25, 20, 15, 10, 5],
    ]

    # rot 180 degrees
    assert dataset._rot180(matrix) == [
        [25, 24, 23, 22, 21],
        [20, 19, 18, 17, 16],
        [15, 14, 13, 12, 11],
        [10, 9, 8, 7, 6],
        [5, 4, 3, 2, 1],
    ]

    # rot 270 degrees
    assert dataset._rot270(matrix) == [
        [5, 10, 15, 20, 25],
        [4, 9, 14, 19, 24],
        [3, 8, 13, 18, 23],
        [2, 7, 12, 17, 22],
        [1, 6, 11, 16, 21],
    ]

    # hmirror
    assert dataset._hmirror(matrix) == [
        [21, 22, 23, 24, 25],
        [16, 17, 18, 19, 20],
        [11, 12, 13, 14, 15],
        [6, 7, 8, 9, 10],
        [1, 2, 3, 4, 5],
    ]

    # vmirror
    assert dataset._vmirror(matrix) == [
        [5, 4, 3, 2, 1],
        [10, 9, 8, 7, 6],
        [15, 14, 13, 12, 11],
        [20, 19, 18, 17, 16],
        [25, 24, 23, 22, 21],
    ]

    # dmirror
    assert dataset._dmirror(matrix) == [
        [1, 6, 11, 16, 21],
        [2, 7, 12, 17, 22],
        [3, 8, 13, 18, 23],
        [4, 9, 14, 19, 24],
        [5, 10, 15, 20, 25],
    ]

    # cmirror
    assert dataset._cmirror(matrix) == [
        [25, 20, 15, 10, 5],
        [24, 19, 14, 9, 4],
        [23, 18, 13, 8, 3],
        [22, 17, 12, 7, 2],
        [21, 16, 11, 6, 1],
    ]

    # map
    assert dataset._map(matrix, a=13, b=0) == [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 0, 14, 15],  # 13 -> 0
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25],
    ]

    # zero divisible
    assert dataset._zero_divisible(matrix, k=3) == [
        [1, 2, 0, 4, 5],
        [0, 7, 8, 0, 10],
        [11, 0, 13, 14, 0],
        [16, 17, 0, 19, 20],
        [0, 22, 23, 0, 25],
    ]

    # crop
    assert dataset._crop(matrix, row_start=2, row_end=4, col_start=1, col_end=3) == [
        [6, 7, 8],
        [11, 12, 13],
        [16, 17, 18],
    ]

    # remove every nth row
    assert dataset._remove_every_nth_row(matrix, n=2) == [
        [1, 2, 3, 4, 5],
        [11, 12, 13, 14, 15],
        [21, 22, 23, 24, 25],
    ]

    # remove every nth col
    assert dataset._remove_every_nth_col(matrix, n=2) == [
        [1, 3, 5],
        [6, 8, 10],
        [11, 13, 15],
        [16, 18, 20],
        [21, 23, 25],
    ]


def test_manipulate_matrix_score_answer():
    """Test the score_answer method"""
    config = ManipulateMatrixConfig(seed=42)
    dataset = ManipulateMatrixDataset(config)

    entry = {"answer": dataset._matrix_to_str([[1, 2, 3], [4, 5, 6], [7, 8, 9]])}

    # perfect match
    answer = "1 2 3\n4 5 6\n7 8 9"
    assert dataset.score_answer(answer, entry) == 1.0

    # model answer contains unnecessary empty spaces
    answer = "1   2 3\n4 5 6   \n7 8 9  "
    assert dataset.score_answer(answer, entry) == 1.0

    # incorrect answer
    answer = "1 2 3\n4 5 6\n7 8 8"
    assert dataset.score_answer(answer, entry) == 0.0

    # answer is none
    answer = None
    assert dataset.score_answer(answer, entry) == 0.0
