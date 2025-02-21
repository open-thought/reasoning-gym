import pytest

from reasoning_gym.arithmetic.bitwise_arithmetic import BitwiseArithmeticConfig, BitwiseArithmeticDataset


def test_bitwise_arithmetic():
    """Test basic properties and solution of generated items"""

    # Easy
    config = BitwiseArithmeticConfig(
        seed=42,
        size=2000,
        difficulty=1,
    )
    dataset = BitwiseArithmeticDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Test the scoring
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0
        assert dataset.score_answer(answer=None, entry=item) == 0.0
        assert dataset.score_answer(answer="kitty cat", entry=item) == 0.01

    config = BitwiseArithmeticConfig(
        seed=42,
        size=100,
        difficulty=5,
    )
    dataset = BitwiseArithmeticDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0

    config = BitwiseArithmeticConfig(
        seed=42,
        size=100,
        difficulty=10,
    )
    dataset = BitwiseArithmeticDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0
