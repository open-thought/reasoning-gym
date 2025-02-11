import pytest

from reasoning_gym.cognition.rectangle_count import RectangleCountConfig, RectangleCountDataset


def test_dice():
    """Test basic properties and solution of generated items"""
    config = RectangleCountConfig(seed=42, size=50, max_rectangles=15, width=40, height=40)
    dataset = RectangleCountDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Test the scoring
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0
        assert dataset.score_answer(answer=None, entry=item) == 0.0
