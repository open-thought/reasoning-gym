"""Tests for Ransom Note questions generation"""

import json

import pytest

from reasoning_gym.algorithmic.ransom_note import RansomNoteConfig, RansomNoteDataset


def test_ransom_note_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = RansomNoteConfig(max_note_length=-1)  # Negative not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = RansomNoteConfig(max_note_length=0)  # Zero not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = RansomNoteConfig(max_magazine_length=-1)  # Negative not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = RansomNoteConfig(max_magazine_length=0)  # Zero not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = RansomNoteConfig(max_magazine_length=1)  # One not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = RansomNoteConfig(
            max_note_length=3, max_magazine_length=2
        )  # max_note_length must be less than max_magazine_length
        config.validate()

    with pytest.raises(AssertionError):
        config = RansomNoteConfig(p_solvable=-0.01)  # p_solvable must be between 0 and 1
        config.validate()

    with pytest.raises(AssertionError):
        config = RansomNoteConfig(p_solvable=1.01)  # p_solvable must be between 0 and 1
        config.validate()


def test_ransom_note_dataset_deterministic():
    """Test that dataset generates same items with same seed"""
    config = RansomNoteConfig(seed=42, size=10)
    dataset1 = RansomNoteDataset(config)
    dataset2 = RansomNoteDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_group_anagrams_dataset_items():
    """Test basic properties of generated items"""
    config = RansomNoteConfig(max_note_length=10, max_magazine_length=30, size=10, seed=42)
    dataset = RansomNoteDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        # Check item structure
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata
        assert "ransom_note" in item["metadata"]
        assert "magazine" in item["metadata"]
        assert "solution" in item["metadata"]
        assert "solvable" in item["metadata"]

        ransom_note = item["metadata"]["ransom_note"]
        magazine = item["metadata"]["magazine"]
        solution = item["metadata"]["solution"]
        solvable = item["metadata"]["solvable"]

        # Verify dimensions
        assert len(ransom_note) <= config.max_note_length
        assert len(ransom_note) <= len(magazine)
        assert len(magazine) <= config.max_magazine_length
        assert solution == solvable

        # Test the scoring
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0
        assert dataset.score_answer(answer="gibberish", entry=item) == 0.01
        assert dataset.score_answer(answer=None, entry=item) == 0.0


def test_ransom_note_dataset_iteration():
    """Test that iteration respects dataset size"""
    config = RansomNoteConfig(size=5, seed=42)
    dataset = RansomNoteDataset(config)

    items = list(dataset)
    assert len(items) == config.size

    # Test multiple iterations yield same items
    assert items == list(dataset)


def test_ransom_note_answer():
    """Test the _can_construct method"""
    config = RansomNoteConfig(seed=42)
    dataset = RansomNoteDataset(config)

    # Correct solution
    ransom_note, magazine = "ab", "badhergh"
    assert dataset._can_construct(ransom_note, magazine) == True

    # Inorrect solution
    ransom_note, magazine = "az", "badhergh"
    assert dataset._can_construct(ransom_note, magazine) == False
