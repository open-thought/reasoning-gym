"""Tests for number sorting task generation"""

import pytest

from reasoning_gym.algorithmic.number_sorting import NumberSortingConfig, NumberSortingCurriculum, NumberSortingDataset


def test_number_sorting_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = NumberSortingConfig(min_numbers=0)
        config.validate()

    with pytest.raises(AssertionError):
        config = NumberSortingConfig(min_numbers=10, max_numbers=5)
        config.validate()

    with pytest.raises(AssertionError):
        config = NumberSortingConfig(min_decimals=-1)
        config.validate()

    with pytest.raises(AssertionError):
        config = NumberSortingConfig(min_value=100, max_value=0)
        config.validate()


def test_number_sorting_dataset_deterministic():
    """Test that dataset generates same items with same seed"""
    config = NumberSortingConfig(seed=42, size=10)
    dataset1 = NumberSortingDataset(config)
    dataset2 = NumberSortingDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_number_sorting_dataset_items():
    """Test basic properties of generated items"""
    config = NumberSortingConfig(
        min_numbers=3, max_numbers=6, min_decimals=1, max_decimals=3, min_value=-10.0, max_value=10.0, size=10, seed=42
    )
    dataset = NumberSortingDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        # Check item structure
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata
        assert "original_numbers" in item["metadata"]
        assert "direction" in item["metadata"]
        assert "sorted_numbers" in item["metadata"]

        # Verify number count constraints
        numbers = item["metadata"]["original_numbers"]
        assert len(numbers) >= config.min_numbers
        assert len(numbers) <= config.max_numbers

        # Verify decimal places
        for num in numbers:
            decimal_places = len(num.split(".")[-1]) if "." in num else 0
            assert decimal_places >= config.min_decimals
            assert decimal_places <= config.max_decimals

        # Verify value range
        for num in numbers:
            value = float(num)
            assert config.min_value <= value <= config.max_value

        # Verify sorting
        direction = item["metadata"]["direction"]
        sorted_numbers = [float(x) for x in eval(item["answer"])]
        if direction == "ascending":
            assert sorted_numbers == sorted(sorted_numbers)
        else:
            assert sorted_numbers == sorted(sorted_numbers, reverse=True)


def test_number_sorting_dataset_iteration():
    """Test that iteration respects dataset size"""
    config = NumberSortingConfig(size=5, seed=42)
    dataset = NumberSortingDataset(config)

    items = list(dataset)
    assert len(items) == config.size

    # Test multiple iterations yield same items
    assert items == list(dataset)


def test_number_sorting_curriculum():
    curriculum = NumberSortingCurriculum()

    base_value = {"size": 150, "seed": 1}

    base_cfg: NumberSortingConfig = curriculum.generate_configuration(base_value)
    assert base_cfg.seed == 1
    assert base_cfg.size == 150
    assert base_cfg.min_numbers == 10 and base_cfg.max_numbers == 50
    assert base_cfg.min_decimals == 0 and base_cfg.max_decimals == 1
    assert base_cfg.min_value == -100 and base_cfg.max_value == 100

    # test incrementing some attribute levels
    curriculum.increment_attr_level("numbers")
    curriculum.increment_attr_level("decimals")
    increased_cfg = curriculum.generate_configuration(base_value)
    assert increased_cfg.min_numbers == 10 and increased_cfg.max_numbers == 100
    assert increased_cfg.min_decimals == 0 and increased_cfg.max_decimals == 2
    assert increased_cfg.min_value == -100 and increased_cfg.max_value == 100

    # test decrementing attribute level for numbers again
    curriculum.decrement_attr_level("numbers")
    partially_decreased_cfg = curriculum.generate_configuration(base_value)
    assert partially_decreased_cfg.min_numbers == 10 and partially_decreased_cfg.max_numbers == 50
    assert partially_decreased_cfg.min_decimals == 0 and partially_decreased_cfg.max_decimals == 2
    assert partially_decreased_cfg.min_value == -100 and partially_decreased_cfg.max_value == 100


def test_number_sorting_score_answer():
    """Test the score_answer method for correctly evaluating model responses."""
    # Create a dataset instance
    config = NumberSortingConfig(seed=42)
    dataset = NumberSortingDataset(config)

    # Create a mock entry similar to the example provided
    mock_entry = {
        "question": "Sort these numbers in ascending order: -16.5, -83.6, -95.7, -97.8, 61.5, 71.08, -92.85",
        "answer": "['-97.8', '-95.7', '-92.8', '-83.6', '-16.5', '61.5', '71.1']",
        "metadata": {
            "direction": "ascending",
            "original_numbers": ["-16.5", "-83.6", "-95.7", "-97.8", "61.5", "71.08", "-92.85"],
            "sorted_numbers": ["-97.8", "-95.7", "-92.8", "-83.6", "-16.5", "61.5", "71.1"],
        },
    }

    # Test case 1: Exact match should score 1.0
    exact_match = "['-97.8', '-95.7', '-92.8', '-83.6', '-16.5', '61.5', '71.1']"
    assert dataset.score_answer(exact_match, mock_entry) == 1.0

    # Test case 2: Answer with small numerical differences but correct order should score 1.0
    close_match = "['-97.8', '-95.7', '-92.85', '-83.6', '-16.5', '61.5', '71.08']"
    assert dataset.score_answer(close_match, mock_entry) == 1.0

    # Test case 3: Incorrectly sorted answer should score 0.0
    wrong_order = "['-16.5', '-83.6', '-92.85', '-95.7', '-97.8', '61.5', '71.08']"
    assert dataset.score_answer(wrong_order, mock_entry) == 0.0

    # Test case 4: Answer with wrong length should score 0.0
    wrong_length = "['-97.8', '-95.7', '-92.85', '-83.6', '-16.5', '61.5']"
    assert dataset.score_answer(wrong_length, mock_entry) == 0.0

    # Test case 5: Non-list answer should score 0.0
    non_list = "'-97.8', '-95.7', '-92.85', '-83.6', '-16.5', '61.5', '71.08'"
    assert dataset.score_answer(non_list, mock_entry) == 0.0

    # Test case 6: None answer should score 0.0
    assert dataset.score_answer(None, mock_entry) == 0.0

    # Test case 7: Correctly sorted but with larger numerical differences (beyond tolerance)
    beyond_tolerance = "['-97.8', '-95.7', '-91.0', '-83.6', '-16.5', '61.5', '72.0']"
    assert dataset.score_answer(beyond_tolerance, mock_entry) == 0.0

    # Test case 8: Descending order test
    descending_entry = {
        "answer": "['71.1', '61.5', '-16.5', '-83.6', '-92.8', '-95.7', '-97.8']",
        "metadata": {
            "direction": "descending",
            "sorted_numbers": ["71.1", "61.5", "-16.5", "-83.6", "-92.8", "-95.7", "-97.8"],
        },
    }
    descending_match = "['71.08', '61.5', '-16.5', '-83.6', '-92.85', '-95.7', '-97.8']"
    assert dataset.score_answer(descending_match, descending_entry) == 1.0
