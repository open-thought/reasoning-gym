import pytest

from reasoning_gym.games.sokoban import SokobanConfig, SokobanDataset


def test_sokoban():
    """Test basic properties and solution of generated items"""

    dataset = SokobanDataset(SokobanConfig(size=10, seed=1234))
    for i, item in enumerate(dataset):
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0

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
        assert dataset.score_answer(answer="RU", entry=item) == 0.0
        assert dataset.score_answer(answer=None, entry=item) == 0.0

    # Hard
    config = SokobanConfig(seed=42, min_h=15, max_h=20, min_w=15, max_w=20, min_boxes=10, max_boxes=15, size=3, max_depth=90)
    dataset = SokobanDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Test the scoring
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0
        assert dataset.score_answer(answer=None, entry=item) == 0.0


    # min == max ranges
    config = SokobanConfig(seed=42, min_h=11, max_h=11, min_w=11, max_w=11, min_boxes=11, max_boxes=11, size=3, max_depth=60)
    dataset = SokobanDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Test the scoring
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0
        assert dataset.score_answer(answer=None, entry=item) == 0.0

