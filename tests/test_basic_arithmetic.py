import pytest

from reasoning_gym.arithmetic.basic_arithmetic import (
    BasicArithmeticDataset,
    BasicArithmeticDatasetConfig,
    eval_floordiv,
)


def test_arithmetic_dataset_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = BasicArithmeticDatasetConfig(min_terms=0)
        config.validate()

    with pytest.raises(AssertionError):
        config = BasicArithmeticDatasetConfig(min_terms=3, max_terms=2)
        config.validate()

    with pytest.raises(AssertionError):
        config = BasicArithmeticDatasetConfig(operators=["^"])  # Invalid operator
        config.validate()


def test_arithmetic_dataset_deterministic():
    """Test that dataset generates same items with same seed"""
    config = BasicArithmeticDatasetConfig(seed=42, size=10)
    dataset1 = BasicArithmeticDataset(config)
    dataset2 = BasicArithmeticDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_arithmetic_dataset_items():
    """Test basic properties of generated items"""
    config = BasicArithmeticDatasetConfig(min_terms=2, max_terms=4, min_digits=1, max_digits=2, size=100, seed=42)
    dataset = BasicArithmeticDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Verify the answer matches the expression
        expression = item["metadata"]["expression"]
        answer = eval_floordiv(expression)  # Safe here as we control the expression
        assert str(answer) == item["answer"]


def test_arithmetic_dataset_format_styles():
    """Test different question format styles"""
    config = BasicArithmeticDatasetConfig(
        size=10,
        seed=42,
        format_style="simple",
        min_terms=2,
        max_terms=3,  # Keep expressions simple for testing
        min_digits=1,
        max_digits=2,
    )
    dataset = BasicArithmeticDataset(config)
    assert all(item["question"].strip().endswith(".") for item in dataset)

    config = BasicArithmeticDatasetConfig(
        size=10,
        seed=42,
        format_style="natural",
        min_terms=2,
        max_terms=3,  # Keep expressions simple for testing
        min_digits=1,
        max_digits=2,
    )
    dataset = BasicArithmeticDataset(config)
    assert all(item["question"].strip().endswith(".") or item["question"].strip().endswith("?") for item in dataset)


def test_arithmetic_dataset_iteration():
    """Test that iteration respects dataset size"""
    config = BasicArithmeticDatasetConfig(min_terms=2, max_terms=2, size=5, seed=42)  # Small size for testing
    dataset = BasicArithmeticDataset(config)

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
