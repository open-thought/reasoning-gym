"""Tests for spell backward task generation"""

import pytest

from reasoning_gym.algorithmic.spell_backward import SpellBackwardConfig, SpellBackwardCurriculum, SpellBackwardDataset


def test_spell_backward_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = SpellBackwardConfig(min_word_len=0)
        config.validate()

    with pytest.raises(AssertionError):
        config = SpellBackwardConfig(min_word_len=4, max_word_len=3)
        config.validate()


def test_spell_backward_dataset_deterministic():
    """Test that dataset generates same items with same seed"""
    config = SpellBackwardConfig(seed=42, size=10)
    dataset1 = SpellBackwardDataset(config)
    dataset2 = SpellBackwardDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_spell_backward_dataset_items():
    """Test basic properties of generated items"""
    config = SpellBackwardConfig(min_word_len=3, size=10, seed=42)
    dataset = SpellBackwardDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        # Check item structure
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata
        assert "word" in item["metadata"]
        assert "word_len" in item["metadata"]

        # Verify word length constraint
        word = item["metadata"]["word"]
        assert len(word) >= config.min_word_len

        # Verify answer is correct
        assert item["answer"] == word[::-1]


def test_spell_backward_dataset_iteration():
    """Test that iteration respects dataset size"""
    config = SpellBackwardConfig(size=5, seed=42)
    dataset = SpellBackwardDataset(config)

    items = list(dataset)
    assert len(items) == config.size

    # Test multiple iterations yield same items
    assert items == list(dataset)


def test_spell_backward_curriculum():
    curriculum = SpellBackwardCurriculum()

    base_value = {"size": 150, "seed": 1}

    base_cfg: SpellBackwardConfig = curriculum.generate_configuration(base_value)
    assert base_cfg.seed == 1
    assert base_cfg.size == 150
    assert base_cfg.min_word_len == 3 and base_cfg.max_word_len == 3

    # test incrementing attribute levels
    curriculum.increment_attr_level("word_len")
    increased_cfg = curriculum.generate_configuration(base_value)
    assert increased_cfg.min_word_len == 3 and increased_cfg.max_word_len == 5

    # test decrementing attribute levels
    curriculum.decrement_attr_level("word_len")
    partially_decreased_cfg = curriculum.generate_configuration(base_value)
    assert partially_decreased_cfg.min_word_len == 3 and partially_decreased_cfg.max_word_len == 3
