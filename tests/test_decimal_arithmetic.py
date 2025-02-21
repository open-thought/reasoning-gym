import pytest

from reasoning_gym.arithmetic.decimal_arithmetic import DecimalArithmeticConfig, DecimalArithmeticDataset


def test_decimal_arithmetic():
    """Test basic properties and solution of generated items"""

    # Easy
    config = DecimalArithmeticConfig(
        seed=42, size=2000, min_num_decimal_places=3, max_num_decimal_places=3, precision=5, terms=3
    )
    dataset = DecimalArithmeticDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Test the scoring
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0

    # M
    config = DecimalArithmeticConfig(
        seed=42, size=2000, min_num_decimal_places=3, max_num_decimal_places=6, precision=8, terms=6
    )
    dataset = DecimalArithmeticDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0

    # H
    config = DecimalArithmeticConfig(
        seed=42, size=2000, min_num_decimal_places=3, max_num_decimal_places=13, precision=15, terms=10
    )
    dataset = DecimalArithmeticDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0
