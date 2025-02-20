from reasoning_gym.arithmetic.decimal_number_comparison import (
    DecimalNumberComparisonConfig,
    DecimalNumberComparisonDataset,
)


def test_decimal_number_comparison():
    """Test basic decimal comparison"""
    config = DecimalNumberComparisonConfig()
    dataset = DecimalNumberComparisonDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Test the scoring
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0
