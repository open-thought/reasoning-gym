"""Tests for Largest Island puzzle generation"""

import pytest

from reasoning_gym.graphs.largest_island import LargestIslandConfig, LargestIslandCurriculum, LargestIslandDataset


def test_largest_island_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = LargestIslandConfig(min_rows=0)  # 0 not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = LargestIslandConfig(min_cols=0)  # 0 not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = LargestIslandConfig(min_rows=10, max_rows=5)  # min > max
        config.validate()

    with pytest.raises(AssertionError):
        config = LargestIslandConfig(min_cols=10, max_cols=5)  # min > max
        config.validate()

    with pytest.raises(AssertionError):
        config = LargestIslandConfig(min_num_islands=-1)  # neg not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = LargestIslandConfig(min_island_size=-1)  # neg not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = LargestIslandConfig(min_num_islands=5, max_num_islands=3)  # min > max
        config.validate()

    with pytest.raises(AssertionError):
        config = LargestIslandConfig(min_island_size=5, max_island_size=3)  # min > max
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
    config = LargestIslandConfig(
        min_rows=5,
        max_rows=10,
        min_cols=5,
        max_cols=10,
        size=10,
        seed=42,
    )
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

        # Verify grid dimensions
        assert 5 <= len(grid) <= 10
        assert all(0 <= len(row) <= 10 for row in grid)


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
    config = LargestIslandConfig(size=5, seed=42)
    dataset = LargestIslandDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        for row in item["metadata"]["grid"]:
            assert all(cell in {0, 1} for cell in row)


def test_largest_island_answer():
    """Test the _get_largest_island method"""
    config = LargestIslandConfig(seed=42)
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


def test_largest_island_curriculum():
    curriculum = LargestIslandCurriculum()

    base_value = {"size": 150, "seed": 1}

    base_cfg: LargestIslandConfig = curriculum.generate_configuration(base_value)
    assert base_cfg.seed == 1
    assert base_cfg.size == 150
    assert base_cfg.min_rows == 5 and base_cfg.max_rows == 5
    assert base_cfg.min_cols == 5 and base_cfg.max_cols == 5
    assert base_cfg.min_num_islands == 2 and base_cfg.max_num_islands == 2
    assert base_cfg.min_island_size == 5 and base_cfg.max_island_size == 5

    # test incrementing attribute levels
    curriculum.increment_attr_level("rows")
    curriculum.increment_attr_level("cols")
    curriculum.increment_attr_level("num_islands")
    curriculum.increment_attr_level("island_size")
    increased_cfg = curriculum.generate_configuration(base_value)
    assert increased_cfg.min_rows == 5 and increased_cfg.max_rows == 25
    assert increased_cfg.min_cols == 5 and increased_cfg.max_cols == 25
    assert increased_cfg.min_num_islands == 2 and increased_cfg.max_num_islands == 5
    assert increased_cfg.min_island_size == 5 and increased_cfg.max_island_size == 10

    # test decrementing attribute level for num_islands again
    curriculum.decrement_attr_level("num_islands")
    partially_decreased_cfg = curriculum.generate_configuration(base_value)
    assert partially_decreased_cfg.min_rows == 5 and partially_decreased_cfg.max_rows == 25
    assert partially_decreased_cfg.min_cols == 5 and partially_decreased_cfg.max_cols == 25
    assert partially_decreased_cfg.min_num_islands == 2 and partially_decreased_cfg.max_num_islands == 2
    assert partially_decreased_cfg.min_island_size == 5 and partially_decreased_cfg.max_island_size == 10
