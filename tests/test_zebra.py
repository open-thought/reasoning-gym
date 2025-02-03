import pytest

from reasoning_gym.logic.zebra_puzzles import ZebraConfig, ZebraDataset


def test_zebra_puzzles():
    """Test basic properties and solution of generated items"""

    config = ZebraConfig(seed=42, size=10, k=4, m=4)
    dataset = ZebraDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Test the scoring
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0
        assert dataset.score_answer(answer=None, entry=item) == 0.0
