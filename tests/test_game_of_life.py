import pytest

from reasoning_gym.games.game_of_life import GameOfLifeConfig, GameOfLifeDataset


def test_game_of_life():
    """Test basic properties and solution of generated items"""

    # Easy
    config = GameOfLifeConfig(seed=42, size=1, grid_size_x=20, grid_size_y=20, filled_cells=10, simulation_steps=1)
    dataset = GameOfLifeDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # # Check metadata contains required fields
        assert "grid_size_x" in item["metadata"]
        assert "grid_size_y" in item["metadata"]
        assert "filled_cells" in item["metadata"]
        assert "simulation_steps" in item["metadata"]

        # # Test the scoring
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0
        assert dataset.score_answer(answer=None, entry=item) == 0.0
