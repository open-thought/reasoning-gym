"""Tests for Shortest Path questions generation"""

import pytest

from reasoning_gym.graphs.shortest_path import ShortestPathConfig, ShortestPathDataset


def test_shortest_path_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = ShortestPathConfig(min_rows=-1)  # Negative not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = ShortestPathConfig(min_rows=0)  # Zero not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = ShortestPathConfig(min_cols=-1)  # Negative not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = ShortestPathConfig(min_cols=0)  # Zero not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = ShortestPathConfig(min_rows=10, max_rows=5)  # min > max
        config.validate()

    with pytest.raises(AssertionError):
        config = ShortestPathConfig(min_cols=10, max_cols=5)
        config.validate()

    with pytest.raises(AssertionError):
        config = ShortestPathConfig(p_blocked=-0.1)
        config.validate()

    with pytest.raises(AssertionError):
        config = ShortestPathConfig(p_blocked=1.1)
        config.validate()


def test_shortest_path_dataset_deterministic():
    """Test that dataset generates same items with same seed"""
    config = ShortestPathConfig(seed=42, size=10)
    dataset1 = ShortestPathDataset(config)
    dataset2 = ShortestPathDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_shortest_path_dataset_items():
    """Test basic properties of generated items"""
    config = ShortestPathConfig(min_rows=3, max_rows=5, min_cols=3, max_cols=5, size=10, seed=42)
    dataset = ShortestPathDataset(config)

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

        # Verify values
        assert len(matrix) >= 3
        assert len(matrix) <= 5
        assert all(len(row) >= 3 for row in matrix)
        assert all(len(row) <= 5 for row in matrix)
        assert any(cell == "*" for row in matrix for cell in row)  # Start cell
        assert any(cell == "#" for row in matrix for cell in row)  # End cell


def test_shortest_path_dataset_iteration():
    """Test that iteration respects dataset size"""
    config = ShortestPathConfig(size=5, seed=42)
    dataset = ShortestPathDataset(config)

    items = list(dataset)
    assert len(items) == config.size

    # Test multiple iterations yield same items
    assert items == list(dataset)


def test_shortest_path_answer():
    """Test the _get_distances method"""
    config = ShortestPathConfig(seed=42)
    dataset = ShortestPathDataset(config)

    # Simple
    matrix = [
        ["X", "X", "X", "X", "X"],
        ["X", "*", "O", "#", "X"],
        ["X", "O", "X", "O", "X"],
    ]
    assert " ".join(dataset._get_answer(matrix)) == "right right"

    # One shot example in prompt
    matrix = [
        ["X", "X", "X", "X", "X"],
        ["X", "*", "O", "O", "X"],
        ["X", "O", "X", "O", "X"],
        ["X", "X", "X", "O", "#"],
    ]
    assert " ".join(dataset._get_answer(matrix)) == "right right down down right"

    # Impossible solution
    matrix = [
        ["X", "X", "X", "X", "X"],
        ["X", "*", "O", "O", "X"],
        ["X", "O", "X", "O", "X"],
        ["X", "X", "X", "X", "#"],
    ]
    assert dataset._get_answer(matrix) == []

    # Multiple valid solutions of same size
    entry = {
        "answer": "right right down down",
        "metadata": {
            "matrix": [
                ["X", "X", "X", "X", "X"],
                ["X", "*", "O", "O", "X"],
                ["X", "O", "X", "O", "X"],
                ["X", "O", "O", "#", "X"],
            ]
        },
    }
    assert dataset.score_answer("right right down down", entry) == 1.0
    assert dataset.score_answer("down down right right", entry) == 1.0

    # Partial solution (valid, but longer than oracle)
    entry = {
        "answer": "right right",
        "metadata": {
            "matrix": [
                ["X", "X", "X", "X", "X"],
                ["X", "*", "O", "#", "X"],
                ["X", "O", "X", "O", "X"],
                ["X", "O", "O", "O", "X"],
            ]
        },
    }
    assert dataset.score_answer("right right", entry) == 1.0
    assert dataset.score_answer("down down right right up up", entry) == 0.5

    # Invalid solution (steps over X)
    entry = {
        "answer": "right right down down",
        "metadata": {
            "matrix": [
                ["X", "X", "X", "X", "X"],
                ["X", "*", "O", "O", "X"],
                ["X", "O", "X", "O", "X"],
                ["X", "O", "O", "#", "X"],
            ]
        },
    }
    assert dataset.score_answer("right down right down", entry) == 0.01

    # Answer is None
    entry = {
        "answer": "right right down down",
        "metadata": {
            "matrix": [
                ["X", "X", "X", "X", "X"],
                ["X", "*", "O", "O", "X"],
                ["X", "O", "X", "O", "X"],
                ["X", "O", "O", "#", "X"],
            ]
        },
    }
    assert dataset.score_answer(None, entry) == 0.0
