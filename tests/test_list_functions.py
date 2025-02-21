from random import Random

import pytest

from reasoning_gym.induction.list_functions import ListFunctionsDataset, ListFunctionsDatasetConfig


def test_list_functions_config_validation():
    """Test that config validation works"""
    config = ListFunctionsDatasetConfig(size=-1)
    with pytest.raises(AssertionError):
        config.validate()


def test_list_functions_deterministic():
    """Test that dataset generates same items with same seed"""
    config = ListFunctionsDatasetConfig(seed=42, size=10)
    dataset1 = ListFunctionsDataset(config)
    dataset2 = ListFunctionsDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_list_functions_items():
    """Test basic properties of generated items"""
    config = ListFunctionsDatasetConfig(size=50, seed=42)
    dataset = ListFunctionsDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert isinstance(item["question"], str)
        assert isinstance(item["answer"], str)


def test_list_functions_iteration():
    """Test that iteration respects dataset size"""
    config = ListFunctionsDatasetConfig(size=5, seed=42)  # Small size for testing
    dataset = ListFunctionsDataset(config)

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


def test_list_functions_generators():
    """Test generator loading and access"""
    config = ListFunctionsDatasetConfig()
    dataset = ListFunctionsDataset(config)

    # Test lazy loading
    assert dataset._generators is None
    _ = dataset.generators  # Access to trigger loading
    assert dataset._generators is not None

    # Test generator mapping
    assert isinstance(dataset.generators, dict)
    assert len(dataset.generators) > 0
    i = 0
    rng = Random(18)
    for key in sorted(dataset.generators.keys()):
        generator = dataset.generators[key]
        assert callable(generator)

        print(i, key)
        for _ in range(10):
            x = generator(rng)
            assert isinstance(x, dict)

        i += 1
