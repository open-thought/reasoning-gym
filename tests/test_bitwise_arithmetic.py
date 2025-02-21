import pytest

from reasoning_gym.arithmetic.bitwise_arithmetic import BitwiseArithmeticConfig, BitwiseArithmeticDataset


def test_bitwise_arithmetic_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = BitwiseArithmeticConfig(difficulty=0)
        config.validate()

    with pytest.raises(AssertionError):
        config = BitwiseArithmeticConfig(difficulty=11)
        config.validate()


def test_bitwise_arithmetic_deterministic():
    """Test that dataset generates same items with same seed"""
    config = BitwiseArithmeticConfig(seed=42, size=10)
    dataset1 = BitwiseArithmeticDataset(config)
    dataset2 = BitwiseArithmeticDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_bitwise_arithmetic_items():
    """Test basic properties of generated items"""
    config = BitwiseArithmeticConfig(difficulty=1, size=100, seed=42)
    dataset = BitwiseArithmeticDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Verify the answer matches the problem
        problem = item["metadata"]["problem"]
        answer = item["answer"]
        assert dataset.score_answer(answer=answer, entry=item) == 1.0

        # Test scoring edge cases
        assert dataset.score_answer(answer=None, entry=item) == 0.0
        assert dataset.score_answer(answer="invalid", entry=item) == 0.01


def test_bitwise_arithmetic_difficulty_levels():
    """Test that different difficulty levels produce appropriate complexity"""
    for difficulty in [1, 2, 3]:
        config = BitwiseArithmeticConfig(difficulty=difficulty, size=50, seed=42)
        dataset = BitwiseArithmeticDataset(config)

        for item in dataset:
            # All items should be valid regardless of difficulty
            assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0

            # Higher difficulty should generally produce more operators
            problem = item["metadata"]["problem"]
            num_operators = sum(1 for c in problem if c in ["+", "-", "*", "<", ">"])

            if difficulty == 1:
                assert num_operators <= 2  # Simple expressions
            elif difficulty >= 5:
                # More complex expressions should exist at higher difficulties
                found_complex = False
                for item in dataset:
                    if sum(1 for c in item["metadata"]["problem"] if c in ["+", "-", "*", "<", ">"]) > 2:
                        found_complex = True
                        break
                assert found_complex


def test_bitwise_arithmetic_iteration():
    """Test that iteration respects dataset size"""
    config = BitwiseArithmeticConfig(difficulty=1, size=5, seed=42)  # Small size for testing
    dataset = BitwiseArithmeticDataset(config)

    # Test manual iteration
    items = []
    for item in dataset:
        items.append(item)
    assert len(items) == config.size, "Iterator should yield exactly size items"

    # Test list conversion
    items = list(dataset)
    assert len(items) == config.size, "Iterator should yield exactly size items"

    # Test multiple iterations
    first_items = list(dataset)
    second_items = list(dataset)
    assert first_items == second_items, "Multiple iterations should yield same items"


def test_bitwise_arithmetic_answer_formats():
    """Test that different answer formats are handled correctly"""
    config = BitwiseArithmeticConfig(difficulty=1, size=10, seed=42)
    dataset = BitwiseArithmeticDataset(config)

    for item in dataset:
        problem = item["metadata"]["problem"]
        correct = item["answer"]

        # Test hex string format
        assert dataset.score_answer(answer=correct, entry=item) == 1.0

        # Test decimal format
        decimal_answer = str(eval(problem))  # Safe as we control the problem
        assert dataset.score_answer(answer=decimal_answer, entry=item) == 1.0

        # Test with "0x" prefix variations
        if correct.startswith("-0x"):
            # For negative numbers, keep the minus sign
            assert dataset.score_answer(answer="-0x" + correct[3:], entry=item) == 1.0
        elif not correct.startswith("0x"):
            # For positive numbers without prefix
            assert dataset.score_answer(answer="0x" + correct, entry=item) == 1.0
