"""Tests for Largest Island puzzle generation"""

import pytest

from reasoning_gym.graphs.largest_island import LargestIslandConfig, LargestIslandDataset


def test_largest_island_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = LargestIslandConfig(rows=-1)  # Negative not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = LargestIslandConfig(rows=0)  # Zero not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = LargestIslandConfig(cols=-1)  # Negative not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = LargestIslandConfig(cols=0)  # Zero not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = LargestIslandConfig(max_num_islands=-1)  # Negative not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = LargestIslandConfig(max_island_size=-1)  # Negative not allowed
        config.validate()


def test_largest_island_dataset_deterministic():
    """Test that dataset generates same items with same seed"""
    config = LargestIslandConfig(seed=42, size=10)
    dataset1 = LargestIslandDataset(config)
    dataset2 = LargestIslandDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_largest_island_dataset_items():
    """Test basic properties of generated items"""
    config = LargestIslandConfig(rows=8, cols=8, max_island_size=5, size=10, seed=42)
    dataset = LargestIslandDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        # Check item structure
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata
        assert "grid" in item["metadata"]
        assert "solution" in item["metadata"]

        grid = item["metadata"]["grid"]
        solution = item["metadata"]["solution"]

        # Verify grid dimensions
        assert len(grid) == 8
        assert all(len(row) == 8 for row in grid)
        assert 0 <= solution <= 5


def test_largest_island_dataset_iteration():
    """Test that iteration respects dataset size"""
    config = LargestIslandConfig(size=5, seed=42)
    dataset = LargestIslandDataset(config)

    items = list(dataset)
    assert len(items) == config.size

    # Test multiple iterations yield same items
    assert items == list(dataset)


def test_largest_island_grid_generation():
    """Test that generated grids are valid"""
    config = LargestIslandConfig(rows=10, cols=10, max_island_size=3, size=5, seed=42)
    dataset = LargestIslandDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        assert item["metadata"]["solution"] <= 3
        for row in item["metadata"]["grid"]:
            assert all(cell in {0, 1} for cell in row)


def test_largest_island_answer():
    """Test the _get_largest_island method"""
    config = LargestIslandConfig(rows=5, cols=5, seed=42)
    dataset = LargestIslandDataset(config)

    grid = [
        [1, 1, 0, 0, 0],
        [1, 0, 0, 0, 1],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1],
    ]
    assert dataset._get_largest_island(grid) == 7

    # Test empty grid
    grid = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    assert dataset._get_largest_island(grid) == 0

    # Test neighboring grids are only horizontally or vertically connected (not diagonally)
    grid = [
        [1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1],
    ]
    assert dataset._get_largest_island(grid) == 9
