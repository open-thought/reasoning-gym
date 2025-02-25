import pytest

from reasoning_gym.games.game_of_life_halting import GameOfLifeHaltingConfig, GameOfLifeHaltingDataset


def test_game_of_life():
    """Test basic properties and solution of generated items"""

    # Easy
    config = GameOfLifeHaltingConfig(
        seed=42, size=10, difficulty=3, grid_size_x=25, grid_size_y=25, max_simulation_steps=25
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

        print(item)
        print(item['question'])

        # # Test the scoring
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0
        assert dataset.score_answer(answer=None, entry=item) == 0.0
