"""Tests for String Synthesis questions generation"""

import pytest

from reasoning_gym.algorithmic.string_synthesis import StringSynthesisConfig, StringSynthesisDataset


def test_string_synthesis_config_validation():
    """Test that invalid configs raise appropriate errors"""

    with pytest.raises(AssertionError):
        config = StringSynthesisConfig(min_initial_blocks=-1)  # Negative not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = StringSynthesisConfig(min_initial_blocks=3, max_initial_blocks=2)  # Min > Max
        config.validate()

    with pytest.raises(AssertionError):
        config = StringSynthesisConfig(max_iterations=0)  # Zero not allowed
        config.validate()


def test_string_synthesis_dataset_deterministic():
    """Test that dataset generates same items with same seed"""
    config = StringSynthesisConfig(seed=42, size=10)
    dataset1 = StringSynthesisDataset(config)
    dataset2 = StringSynthesisDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_string_synthesis_dataset_items():
    """Test basic properties of generated items"""
    config = StringSynthesisConfig(min_initial_blocks=1, max_initial_blocks=3, size=10, seed=42)
    dataset = StringSynthesisDataset(config)

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
        assert len(states) >= 1
        first_state = states[0]
        assert len(first_state) == 9
        for i in range(3):
            assert 0 <= first_state[i] <= 3
        for i in range(3, 9):
            assert first_state[i] == 0
        assert solution == states[-1]
        for i in range(9):
            assert 0 <= solution[i]


def test_string_synthesis_dataset_iteration():
    """Test that iteration respects dataset size"""
    config = StringSynthesisConfig(size=5, seed=42)
    dataset = StringSynthesisDataset(config)

    items = list(dataset)
    assert len(items) == config.size

    # Test multiple iterations yield same items
    assert items == list(dataset)


def test_string_synthesis_answer():
    """Test the _get_answer method"""
    config = StringSynthesisConfig(seed=42)
    dataset = StringSynthesisDataset(config)

    # Empty input
    counts = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert dataset._apply_rule(counts) == [0, 0, 0, 0, 0, 0, 0, 0, 0]

    # Rule 1
    counts = [1, 1, 1, 0, 0, 0, 0, 0, 0]
    assert dataset._apply_rule(counts) == [0, 0, 0, 1, 0, 0, 0, 0, 0]

    # Rule 2
    counts = [1, 1, 0, 0, 0, 0, 0, 0, 0]
    assert dataset._apply_rule(counts) == [0, 0, 0, 0, 0, 1, 0, 0, 0]

    # Rule 3
    counts = [0, 1, 1, 0, 0, 0, 0, 0, 0]
    assert dataset._apply_rule(counts) == [0, 0, 0, 0, 1, 0, 0, 0, 0]

    # Rule 4
    counts = [0, 0, 2, 0, 0, 0, 0, 0, 0]
    assert dataset._apply_rule(counts) == [0, 0, 0, 0, 0, 1, 0, 0, 0]

    # Rule 5
    counts = [0, 0, 0, 1, 0, 1, 0, 0, 0]
    assert dataset._apply_rule(counts) == [0, 0, 0, 0, 0, 0, 1, 1, 0]

    # Rule 6
    counts = [0, 0, 0, 0, 2, 0, 0, 0, 0]
    assert dataset._apply_rule(counts) == [0, 0, 0, 0, 0, 0, 0, 0, 1]

    # 1-shot example provided in the prompt
    A_square, B_square, C_square = 2, 3, 3
    assert dataset._get_answer(A_square, B_square, C_square) == [
        [2, 3, 3, 0, 0, 0, 0, 0, 0],  # Initial state
        [1, 2, 2, 1, 0, 0, 0, 0, 0],  # Rule 1
        [0, 1, 1, 2, 0, 0, 0, 0, 0],  # Rule 1 again
        [0, 0, 0, 2, 1, 0, 0, 0, 0],  # Rule 3 (final state)
    ]
