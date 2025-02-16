"""Tests for word sorting task generation"""

import pytest

from reasoning_gym.algorithmic.word_sorting import TextTransformation, WordSortingConfig, WordSortingDataset


def test_word_sorting_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = WordSortingConfig(min_words=0)
        config.validate()

    with pytest.raises(AssertionError):
        config = WordSortingConfig(min_words=10, max_words=5)
        config.validate()

    with pytest.raises(AssertionError):
        config = WordSortingConfig(min_word_length=0)
        config.validate()

    with pytest.raises(AssertionError):
        config = WordSortingConfig(min_word_length=10, max_word_length=5)
        config.validate()


def test_word_sorting_dataset_deterministic():
    """Test that dataset generates same items with same seed"""
    config = WordSortingConfig(seed=42, size=10)
    dataset1 = WordSortingDataset(config)
    dataset2 = WordSortingDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_word_sorting_transformations():
    """Test different text transformations"""
    seed = 42
    size = 5

    # Test LOWERCASE
    config = WordSortingConfig(transformation=TextTransformation.LOWERCASE, seed=seed, size=size)
    dataset = WordSortingDataset(config)
    for item in dataset:
        for word in item["metadata"]["transformed_words"]:
            if word.isalpha():  # Only test alphabetic strings
                assert word.islower()

    # Test UPPERCASE
    config = WordSortingConfig(transformation=TextTransformation.UPPERCASE, seed=seed, size=size)
    dataset = WordSortingDataset(config)
    for item in dataset:
        for word in item["metadata"]["transformed_words"]:
            if word.isalpha():  # Only test alphabetic strings
                assert word.isupper()

    # Test ORIGINAL
    config = WordSortingConfig(transformation=TextTransformation.ORIGINAL, seed=seed, size=size)
    dataset = WordSortingDataset(config)
    for item in dataset:
        assert item["metadata"]["original_words"] == item["metadata"]["transformed_words"]


def test_word_sorting_dataset_items():
    """Test basic properties of generated items"""
    config = WordSortingConfig(min_words=3, max_words=6, min_word_length=3, max_word_length=8, size=10, seed=42)
    dataset = WordSortingDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        # Check item structure
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata
        assert "original_words" in item["metadata"]
        assert "transformed_words" in item["metadata"]
        assert "direction" in item["metadata"]
        assert "transformation" in item["metadata"]
        assert "sorted_words" in item["metadata"]

        # Verify word count constraints
        words = item["metadata"]["transformed_words"]
        assert len(words) >= config.min_words
        assert len(words) <= config.max_words

        # Verify word length constraints
        for word in words:
            assert len(word) >= config.min_word_length
            assert len(word) <= config.max_word_length

        # Verify sorting
        direction = item["metadata"]["direction"]
        sorted_words = item["answer"].split(", ")
        if direction == "ascending":
            assert sorted_words == sorted(sorted_words)
        else:
            assert sorted_words == sorted(sorted_words, reverse=True)

        # Test the scoring
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0
        assert dataset.score_answer(answer="gibberish", entry=item) == 0.01
        assert dataset.score_answer(answer=None, entry=item) == 0.0


def test_word_sorting_dataset_iteration():
    """Test that iteration respects dataset size"""
    config = WordSortingConfig(size=5, seed=42)
    dataset = WordSortingDataset(config)

    items = list(dataset)
    assert len(items) == config.size

    # Test multiple iterations yield same items
    assert items == list(dataset)


def test_word_sorting_scoring():
    """Test scoring function"""
    config = WordSortingConfig(size=1, seed=42)
    dataset = WordSortingDataset(config)

    item = {
        "metadata": {
            "sorted_words": ["apple", "banana", "cherry"],
        }
    }

    # Correct answer
    answer = "apple, banana, cherry"
    assert dataset.score_answer(answer, item) == 1.0

    # Correct answer, with incorrect spaces
    answer = "apple,banana,        cherry"
    assert dataset.score_answer(answer, item) == 1.0

    # All words present, but not sorted
    answer = "banana, cherry, apple"
    assert dataset.score_answer(answer, item) == 0.2

    # Garbage
    answer = "gibberish"
    assert dataset.score_answer(answer, item) == 0.01

    # Empty answer
    answer = None
    assert dataset.score_answer(answer, item) == 0.0
