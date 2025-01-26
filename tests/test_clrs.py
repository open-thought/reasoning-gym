import pytest
from reasoning_gym.algorithmic.clrs_task import ClrsConfig, ClrsDataset


def test_clrs_dataset_basic():
    dataset_size = 11
    config = ClrsConfig(
        subtask="heapsort",
        min_length=4,
        max_length=4,
        seed=123,
        size=dataset_size,
    )

    dataset = ClrsDataset(config)

    assert len(dataset) == dataset_size


def test_clrs_dataset_items():
    config = ClrsConfig(
        subtask="heapsort",
        min_length=4,
        max_length=5,
        seed=42,
        size=3,
    )
    dataset = ClrsDataset(config)

    for item in dataset:
        # Check keys
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata contents
        metadata = item["metadata"]
        assert "length" in metadata
        assert "arrays" in metadata

        # The question should contain "heapsort:"
        assert "heapsort:" in item["question"]

        # answer is a string
        assert isinstance(item["answer"], str)

        # If there's a "key: [ ... ]" line, ensure it is in arrays
        if "key" in metadata["arrays"]:
            # Confirm that those numeric values appear in the question
            for val in metadata["arrays"]["key"]:
                assert str(val) in item["question"]


def test_clrs_dataset_deterministic():
    config = ClrsConfig(
        "heapsort",
        min_length=4,
        max_length=4,
        seed=999,
        size=2,
    )
    dataset1 = ClrsDataset(config)
    dataset2 = ClrsDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i], "Deterministic mismatch with same config"
