import pytest

from reasoning_gym.games.sokoban import SokobanConfig, SokobanDataset


def test_sokoban():
    """Test basic properties and solution of generated items"""

    # Easy
    config = SokobanConfig(seed=42, size=20)
    dataset = SokobanDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Test the scoring
        assert dataset.score_answer(answer=item["metadata"]["possible_answer"], entry=item) == 1.0
        assert dataset.score_answer(answer=None, entry=item) == 0.0