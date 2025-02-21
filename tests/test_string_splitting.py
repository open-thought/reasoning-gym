"""Tests for String Splitting questions generation"""

import pytest

from reasoning_gym.algorithmic.string_splitting import StringSplittingConfig, StringSplittingDataset


def test_string_splitting_config_validation():
    """Test that invalid configs raise appropriate errors"""

    with pytest.raises(AssertionError):
        config = StringSplittingConfig(min_initial_machines=-1)  # negative not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = StringSplittingConfig(min_initial_machines=3, max_initial_machines=2)  # min > max
        config.validate()


def test_string_splitting_dataset_deterministic():
    """Test that dataset generates same items with same seed"""
    config = StringSplittingConfig(seed=42, size=10)
    dataset1 = StringSplittingDataset(config)
    dataset2 = StringSplittingDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_string_splitting_dataset_items():
    """Test basic properties of generated items"""
    config = StringSplittingConfig(min_initial_machines=1, max_initial_machines=5, size=10, seed=42)
    dataset = StringSplittingDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        # Check item structure
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata
        assert "states" in item["metadata"]
        assert "solution" in item["metadata"]

        states = item["metadata"]["states"]
        solution = item["metadata"]["solution"]

        # Verify dimensions
        assert len(states) > 0
        assert states[-1] == solution
        for i in range(3):
            assert 1 <= states[0][i] <= 5
        for i in range(3, 6):
            assert states[0][i] == 0


def test_string_splitting_dataset_iteration():
    """Test that iteration respects dataset size"""
    config = StringSplittingConfig(size=5, seed=42)
    dataset = StringSplittingDataset(config)

    items = list(dataset)
    assert len(items) == config.size

    # Test multiple iterations yield same items
    assert items == list(dataset)


def test_string_splitting_answer():
    """Test the answer calculation"""
    config = StringSplittingConfig(seed=42)
    dataset = StringSplittingDataset(config)

    # Empty input
    counts = [0, 0, 0, 0, 0, 0]
    assert dataset._apply_rule(counts) == [0, 0, 0, 0, 0, 0]

    # Rule 1: 1A -> 2X 1Y
    counts = [1, 0, 0, 0, 0, 0]
    assert dataset._apply_rule(counts) == [0, 0, 0, 2, 1, 0]

    # Rule 2: 2B -> 1X
    counts = [0, 2, 0, 0, 0, 0]
    assert dataset._apply_rule(counts) == [0, 0, 0, 1, 0, 0]

    # Rule 3: 2C -> 1Y
    counts = [0, 0, 2, 0, 0, 0]
    assert dataset._apply_rule(counts) == [0, 0, 0, 0, 1, 0]

    # Rule 4: B + C -> A
    counts = [0, 1, 1, 0, 0, 0]
    assert dataset._apply_rule(counts) == [1, 0, 0, 0, 0, 0]

    # Rule 5: X + Y -> Z
    counts = [0, 0, 0, 1, 1, 0]
    assert dataset._apply_rule(counts) == [0, 0, 0, 0, 0, 1]

    # 1-shot example used in the prompt
    A_machine, B_machine, C_machine = 2, 0, 1
    assert dataset._get_answer(A_machine, B_machine, C_machine) == [
        [2, 0, 1, 0, 0, 0],
        [1, 0, 1, 2, 1, 0],
        [0, 0, 1, 4, 2, 0],
        [0, 0, 1, 3, 1, 1],
        [0, 0, 1, 2, 0, 2],
    ]
