"""Tests for letter jumbling task generation"""

from random import Random

import pytest

from reasoning_gym.algorithmic.letter_jumble import LetterJumbleConfig, LetterJumbleDataset


def test_letter_jumble_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = LetterJumbleConfig(min_word_len=0)
        config.validate()

    with pytest.raises(AssertionError):
        config = LetterJumbleConfig(min_words=10, max_words=5)
        config.validate()

    with pytest.raises(AssertionError):
        config = LetterJumbleConfig(min_corruption_level=-0.1)
        config.validate()

    with pytest.raises(AssertionError):
        config = LetterJumbleConfig(max_corruption_level=1.1)
        config.validate()


def test_letter_jumble_deterministic():
    """Test that dataset generates same items with same seed"""
    config = LetterJumbleConfig(seed=42, size=10)
    dataset1 = LetterJumbleDataset(config)
    dataset2 = LetterJumbleDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_letter_jumble_scrambling():
    """Test the word scrambling logic"""
    config = LetterJumbleConfig(
        min_word_len=4,
        max_word_len=8,
        min_words=1,
        max_words=1,
        min_corruption_level=0.5,
        max_corruption_level=0.5,
        size=1,
        seed=42,
    )
    dataset = LetterJumbleDataset(config)

    # Test with known word
    word = "testing"
    rng = Random(42)
    scrambled = dataset._scramble_word(word, 0.5, rng)

    # Verify scrambled word:
    # - Has same length as original
    assert len(scrambled) == len(word)
    # - Contains same characters
    assert sorted(scrambled) == sorted(word)
    # - Is different from original (with high probability given 0.5 corruption)
    assert scrambled != word


def test_letter_jumble_dataset_items():
    """Test basic properties of generated items"""
    config = LetterJumbleConfig(
        min_word_len=4,
        max_word_len=8,
        min_words=3,
        max_words=5,
        min_corruption_level=0.1,
        max_corruption_level=0.3,
        size=50,
        seed=42,
    )
    dataset = LetterJumbleDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]

        # Check item structure
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata
        metadata = item["metadata"]
        assert "num_words" in metadata
        assert "corruption_level" in metadata
        assert "scrambled_words" in metadata
        assert "original_words" in metadata

        # Verify word counts
        num_words = metadata["num_words"]
        assert config.min_words <= num_words <= config.max_words
        assert len(metadata["scrambled_words"]) == num_words
        assert len(metadata["original_words"]) == num_words

        # Verify corruption level
        assert config.min_corruption_level <= metadata["corruption_level"] <= config.max_corruption_level

        # Verify word properties
        for word in metadata["original_words"]:
            assert config.min_word_len <= len(word) <= config.max_word_len
            assert word.isalpha()


def test_letter_jumble_iteration():
    """Test that iteration respects dataset size"""
    config = LetterJumbleConfig(size=5, seed=42)
    dataset = LetterJumbleDataset(config)

    items = list(dataset)
    assert len(items) == config.size

    # Test multiple iterations yield same items
    assert items == list(dataset)
