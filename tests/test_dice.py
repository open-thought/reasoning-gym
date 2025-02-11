import pytest

from reasoning_gym.arithmetic.dice import DiceConfig, DiceDataset


def test_dice():
    """Test basic properties and solution of generated items"""
    config = DiceConfig(seed=42, size=50, num_dice=8, max_dice_size=24)
    dataset = DiceDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Test the scoring
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0
        assert dataset.score_answer(answer=None, entry=item) == 0.0

    # Easy
    config = DiceConfig(seed=42, size=1, num_dice=1, max_dice_size=2)
    dataset = DiceDataset(config)

    for item in dataset:
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0
        assert dataset.score_answer(answer=None, entry=item) == 0.0

    # Hard
    config = DiceConfig(seed=42, size=1, num_dice=40, max_dice_size=40)
    dataset = DiceDataset(config)

    for item in dataset:
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0
        assert dataset.score_answer(answer=None, entry=item) == 0.0
