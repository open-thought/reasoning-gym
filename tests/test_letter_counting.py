"""Tests for letter counting task generation"""

import pytest

from reasoning_gym.algorithmic.letter_counting import LetterCountingConfig, LetterCountingDataset


def test_letter_counting_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = LetterCountingConfig(min_words=0)
        config.validate()

    with pytest.raises(AssertionError):
        config = LetterCountingConfig(min_words=10, max_words=5)
        config.validate()


def test_letter_counting_dataset_deterministic():
    """Test that dataset generates same items with same seed"""
    config = LetterCountingConfig(seed=42, size=10)
    dataset1 = LetterCountingDataset(config)
    dataset2 = LetterCountingDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_letter_counting_dataset_items():
    """Test basic properties of generated items"""
    config = LetterCountingConfig(min_words=3, max_words=6, size=10, seed=42)
    dataset = LetterCountingDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        # Check item structure
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata
        assert "span_length" in item["metadata"]
        assert "target_letter" in item["metadata"]
        assert "span" in item["metadata"]

        # Verify span length constraints
        span = item["metadata"]["span"]
        assert len(span) >= config.min_words
        assert len(span) <= config.max_words

        # Verify letter counting
        target_letter = item["metadata"]["target_letter"]
        count = sum(word.lower().count(target_letter) for word in span)
        assert str(count) == item["answer"]


def test_letter_counting_dataset_iteration():
    """Test that iteration respects dataset size"""
    config = LetterCountingConfig(size=5, seed=42)
    dataset = LetterCountingDataset(config)

    items = list(dataset)
    assert len(items) == config.size

    # Test multiple iterations yield same items
    assert items == list(dataset)


def test_letter_counting_text_preprocessing():
    """Test that text preprocessing handles edge cases"""
    config = LetterCountingConfig(size=1, seed=42)
    dataset = LetterCountingDataset(config)

    # Verify words were extracted from text
    assert len(dataset.words) > 0
    # Verify words contain only word characters
    assert all(word.isalnum() for word in dataset.words)
