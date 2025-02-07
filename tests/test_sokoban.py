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
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0
        assert dataset.score_answer(answer="RU", entry=item) == 0.1
        assert dataset.score_answer(answer=None, entry=item) == 0.0

    # Medium
    config = SokobanConfig(seed=42, min_h=40, max_h=50, min_w=40, max_w=50, min_boxes=20, max_boxes=30, size=3)
    dataset = SokobanDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Test the scoring
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0
        assert dataset.score_answer(answer=None, entry=item) == 0.0

    # Hard
    config = SokobanConfig(seed=42, min_h=400, max_h=500, min_w=400, max_w=500, min_boxes=50, max_boxes=50, size=1)
    dataset = SokobanDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Test the scoring
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0
        assert dataset.score_answer(answer=None, entry=item) == 0.0
