import pytest

from reasoning_gym.logic.zebra_puzzles import ZebraConfig, ZebraDataset


def test_zebra_deterministic():
    """Test that dataset generates same items with same seed"""
    config = ZebraConfig(seed=42, size=10, num_people=4, num_characteristics=4)
    dataset1 = ZebraDataset(config)
    dataset2 = ZebraDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_zebra_puzzles():
    """Test basic properties and solution of generated items"""
    config = ZebraConfig(seed=42, size=10, num_people=4, num_characteristics=4)
    dataset = ZebraDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Test the scoring
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0
        assert dataset.score_answer(answer=None, entry=item) == 0.0
