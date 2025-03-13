import json

import pytest

from reasoning_gym.algorithmic.game_of_life import GameOfLifeConfig, GameOfLifeCurriculum, GameOfLifeDataset


def test_game_of_life_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = GameOfLifeConfig(grid_size_x=2)  # Too small
        config.validate()

    with pytest.raises(AssertionError):
        config = GameOfLifeConfig(grid_size_y=1000)  # Too large
        config.validate()

    with pytest.raises(AssertionError):
        config = GameOfLifeConfig(grid_size_x=5, grid_size_y=5, filled_cells=26)  # Too many cells
        config.validate()


def test_game_of_life_deterministic():
    """Test that dataset generates same items with same seed"""
    config = GameOfLifeConfig(seed=42, size=10)
    config2 = GameOfLifeConfig(seed=43, size=10)
    dataset1 = GameOfLifeDataset(config)
    dataset2 = GameOfLifeDataset(config)
    dataset3 = GameOfLifeDataset(config2)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]
        assert dataset1[i] != dataset3[i]


def test_game_of_life_basic_properties():
    """Test basic properties and solution of generated items"""
    config = GameOfLifeConfig(seed=42, size=10, grid_size_x=20, grid_size_y=20, filled_cells=200, simulation_steps=1)
    dataset = GameOfLifeDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata contains required fields
        assert "grid_size_x" in item["metadata"]
        assert "grid_size_y" in item["metadata"]
        assert "filled_cells" in item["metadata"]
        assert "simulation_steps" in item["metadata"]

        # Test the scoring
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0
        assert dataset.score_answer(answer=None, entry=item) == 0.0
        assert dataset.score_answer(answer="invalid json", entry=item) == 0.0

    config = GameOfLifeConfig(seed=43, size=1, grid_size_x=3, grid_size_y=3, filled_cells=1, simulation_steps=1)
    dataset = GameOfLifeDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        ja = json.loads(item["answer"])
        ja[0][0] = 1
        ja[0][1] = 1
        ja[0][2] = 1
        jas = json.dumps(ja)

        # Test the scoring
        assert 0.1 < dataset.score_answer(answer=jas, entry=item) < 1.0


def test_game_of_life_iteration():
    """Test that iteration respects dataset size"""
    config = GameOfLifeConfig(size=5, seed=42)  # Small size for testing
    dataset = GameOfLifeDataset(config)

    # Test manual iteration
    items = []
    for item in dataset:
        items.append(item)
    assert len(items) == config.size, "Iterator should yield exactly size items"

    # Test list conversion
    items = list(dataset)
    assert len(items) == config.size, "Iterator should yield exactly size items"

    # Test multiple iterations
    first_items = list(dataset)
    second_items = list(dataset)
    assert first_items == second_items, "Multiple iterations should yield same items"


def test_game_of_life_curriculum():
    """Test the curriculum for complex arithmetic."""
    curriculum = GameOfLifeCurriculum()
    base_value = {"size": 150, "seed": 1}

    base_cfg: GameOfLifeCurriculum = curriculum.generate_configuration(base_value)

    assert base_cfg.seed == 1
    assert base_cfg.size == 150
    assert base_cfg.grid_size_x == 10
    assert base_cfg.grid_size_y == 10
    assert base_cfg.filled_cells <= base_cfg.grid_size_x * base_cfg.grid_size_y
    assert base_cfg.simulation_steps == 1

    # Test and validate increase in levels
    curriculum.increment_attr_level("grid_size_x")
    curriculum.increment_attr_level("grid_size_y")
    curriculum.increment_attr_level("filled_cells_weights")
    curriculum.increment_attr_level("simulation_steps")

    increased_cfg: GameOfLifeCurriculum = curriculum.generate_configuration(base_value)
    assert increased_cfg.grid_size_x == 100
    assert increased_cfg.grid_size_y == 100
    assert increased_cfg.filled_cells_weights == 0.2
    assert increased_cfg.filled_cells <= increased_cfg.grid_size_x * increased_cfg.grid_size_y
    assert increased_cfg.simulation_steps == 2

    # Test and validate decrease in levels
    curriculum.decrement_attr_level("grid_size_x")
    curriculum.decrement_attr_level("grid_size_y")
    curriculum.decrement_attr_level("filled_cells_weights")
    curriculum.decrement_attr_level("simulation_steps")

    decreased_cfg: GameOfLifeCurriculum = curriculum.generate_configuration(base_value)
    assert decreased_cfg.grid_size_x == 10
    assert decreased_cfg.grid_size_y == 10
    assert decreased_cfg.filled_cells_weights == 0.1
    assert decreased_cfg.filled_cells <= decreased_cfg.grid_size_x * decreased_cfg.grid_size_y
    assert decreased_cfg.simulation_steps == 1

    # Test upper bound boundary condition
    for _ in range(10):
        curriculum.increment_attr_level("grid_size_x")
        curriculum.increment_attr_level("grid_size_y")
        curriculum.increment_attr_level("filled_cells_weights")
        curriculum.increment_attr_level("simulation_steps")
    upper_bound_cfg: GameOfLifeCurriculum = curriculum.generate_configuration(base_value)
    assert upper_bound_cfg.grid_size_x == 10000
    assert upper_bound_cfg.grid_size_y == 10000
    assert upper_bound_cfg.filled_cells_weights == 0.8
    assert upper_bound_cfg.filled_cells <= upper_bound_cfg.grid_size_x * upper_bound_cfg.grid_size_y
    assert upper_bound_cfg.simulation_steps == 10

    # Test lower bound boundary condition
    for _ in range(10):
        curriculum.decrement_attr_level("grid_size_x")
        curriculum.decrement_attr_level("grid_size_y")
        curriculum.decrement_attr_level("filled_cells_weights")
        curriculum.decrement_attr_level("simulation_steps")
    lower_bound_cfg: GameOfLifeCurriculum = curriculum.generate_configuration(base_value)
    assert lower_bound_cfg.grid_size_x == 10
    assert lower_bound_cfg.grid_size_y == 10
    assert lower_bound_cfg.filled_cells_weights == 0.1
    assert lower_bound_cfg.filled_cells <= lower_bound_cfg.grid_size_x * lower_bound_cfg.grid_size_y
    assert lower_bound_cfg.simulation_steps == 1
