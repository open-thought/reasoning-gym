import pytest

from reasoning_gym.arithmetic.basic_arithmetic import BasicArithmeticDataset, BasicArithmeticDatasetConfig
from reasoning_gym.dataset import ReseedingDataset


def test_reseeding_dataset_iteration():
    """Test that ReseedingDataset provides infinite iteration with consistent chunks"""

    # Create base dataset
    config = BasicArithmeticDatasetConfig(
        min_terms=2, max_terms=3, min_digits=1, max_digits=2, operators=["+"], allow_parentheses=False, seed=42, size=10
    )
    base_dataset = BasicArithmeticDataset(config)

    # Create reseeding dataset with small chunk size
    chunk_size = 3
    infinite_dataset = ReseedingDataset(base_dataset, chunk_size=chunk_size)

    # Get first 10 items
    first_items = []
    for _, item in zip(range(10), infinite_dataset):
        first_items.append(item["question"])

    # Create new iterator and verify first 10 items are identical
    second_items = []
    for _, item in zip(range(10), infinite_dataset):
        second_items.append(item["question"])

    assert first_items == second_items, "Items should be deterministic across iterations"

    # Verify chunks are different
    chunk1 = first_items[:chunk_size]
    chunk2 = first_items[chunk_size : 2 * chunk_size]
    assert chunk1 != chunk2, "Different chunks should generate different items"

    # Test score_answer forwarding
    test_item = next(iter(infinite_dataset))
    assert infinite_dataset.score_answer("wrong", test_item) == 0.01
    assert infinite_dataset.score_answer(test_item["answer"], test_item) == 1.0
