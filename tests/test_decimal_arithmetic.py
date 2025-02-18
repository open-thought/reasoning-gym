import pytest

from reasoning_gym.arithmetic.decimal_arithmetic import DecimalArithmeticDataset, DecimalArithmeticDatasetConfig


def test_decimal_arithmetic():
    """Test basic properties and solution of generated items"""

    # Easy
    config = DecimalArithmeticDatasetConfig(seed=42, size=2000, num_decimal_places=3)
    dataset = DecimalArithmeticDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Test the scoring
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0

    # M
    config = DecimalArithmeticDatasetConfig(seed=42, size=2000, num_decimal_places=8)
    dataset = DecimalArithmeticDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0

    # H
    config = DecimalArithmeticDatasetConfig(seed=42, size=2000, num_decimal_places=15)
    dataset = DecimalArithmeticDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0
