import pytest

from reasoning_gym.algorithmic.game_of_life_halting import (
    GameOfLifeHaltingConfig,
    GameOfLifeHaltingCurriculum,
    GameOfLifeHaltingDataset,
)


def test_game_of_life():
    """Test basic properties and solution of generated items"""

    # Easy
    config = GameOfLifeHaltingConfig(
        seed=42, size=10, difficulty=3, grid_size_x=25, grid_size_y=25, max_simulation_steps=99
    )
    dataset = GameOfLifeHaltingDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # # Check metadata contains required fields
        assert "grid_size_x" in item["metadata"]
        assert "grid_size_y" in item["metadata"]

        # # Test the scoring
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0
        assert dataset.score_answer(answer=None, entry=item) == 0.0


def test_game_of_life_halting_deterministic():
    """Test that dataset generates same items with same seed"""
    config = GameOfLifeHaltingConfig(seed=42, size=10)
    config2 = GameOfLifeHaltingConfig(seed=43, size=10)
    dataset1 = GameOfLifeHaltingDataset(config)
    dataset2 = GameOfLifeHaltingDataset(config)
    dataset3 = GameOfLifeHaltingDataset(config2)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]
        assert dataset1[i] != dataset3[i]


def test_game_of_life_halting_curriculum():
    """Test the curriculum for complex arithmetic."""
    curriculum = GameOfLifeHaltingCurriculum()
    base_value = {"size": 150, "seed": 1}

    base_cfg: GameOfLifeHaltingCurriculum = curriculum.generate_configuration(base_value)

    assert base_cfg.seed == 1
    assert base_cfg.size == 150
    assert base_cfg.grid_size_x == 10
    assert base_cfg.grid_size_y == 10
    assert base_cfg.difficulty == 1
    assert base_cfg.num_oscillators == 3
    assert base_cfg.max_simulation_steps == 20

    # Test and validate increase in levels
    curriculum.increment_attr_level("grid_size_x")
    curriculum.increment_attr_level("grid_size_y")
    curriculum.increment_attr_level("difficulty")
    curriculum.increment_attr_level("num_oscillators")
    curriculum.increment_attr_level("max_simulation_steps")

    increased_cfg: GameOfLifeHaltingCurriculum = curriculum.generate_configuration(base_value)
    assert increased_cfg.grid_size_x == 25
    assert increased_cfg.grid_size_y == 25
    assert increased_cfg.difficulty == 2
    assert increased_cfg.num_oscillators == 7
    assert increased_cfg.max_simulation_steps == 50
