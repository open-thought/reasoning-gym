import pytest

from reasoning_gym.algorithmic.game_of_life_halting import GameOfLifeHaltingConfig, GameOfLifeHaltingDataset


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
