from random import Random

import pytest

from reasoning_gym.coaching.base_curriculum import DefaultCurriculumContext, RangeAttributeMode
from reasoning_gym.games.emoji_mystery import EmojiMysteryConfig, EmojiMysteryCurriculum, EmojiMysteryDataset


def test_emoji_mystery_config_validation():
    """Test that config validation works"""
    config = EmojiMysteryConfig(size=-1)
    with pytest.raises(AssertionError):
        config.validate()


def test_emoji_mystery_deterministic():
    """Test that dataset generates same items with same seed"""
    config = EmojiMysteryConfig(seed=42, size=10)
    dataset1 = EmojiMysteryDataset(config)
    dataset2 = EmojiMysteryDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_emoji_mystery_items():
    """Test basic properties of generated items"""
    config = EmojiMysteryConfig(size=100, seed=42)
    dataset = EmojiMysteryDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert isinstance(item["question"], str)
        assert isinstance(item["answer"], str)


def test_emoji_mystery_iteration():
    """Test that iteration respects dataset size"""
    config = EmojiMysteryConfig(size=5, seed=42)  # Small size for testing
    dataset = EmojiMysteryDataset(config)

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


def test_emoji_mystery_encoding_decoding():
    """Test the encoding and decoding functionality"""
    config = EmojiMysteryConfig()
    dataset = EmojiMysteryDataset(config)

    # Test with a simple sentence
    test_sentence = "Hello, World!"
    test_emoji = "😀"

    # Test encoding
    encoded = dataset.encode(test_sentence, test_emoji)
    assert encoded.startswith(test_emoji)

    # Test decoding
    decoded = dataset.decode(encoded)
    assert decoded == test_sentence

    # Test with various sentences
    test_cases = [
        "A simple test.",
        "More complex sentence with numbers 123!",
        "Special characters: @#$%^&*()",
    ]

    for sentence in test_cases:
        encoded = dataset.encode(sentence, test_emoji)
        decoded = dataset.decode(encoded)
        assert decoded == sentence


def test_emoji_mystery_scoring():
    """Test the scoring functionality"""
    config = EmojiMysteryConfig()
    dataset = EmojiMysteryDataset(config)

    # Test exact match
    entry = {"answer": "Test answer"}
    assert dataset.score_answer("Test answer", entry) == 1.0

    # Test partial match
    assert dataset.score_answer("Test answe", entry) == 0.01  # Different length

    # Test None answer
    assert dataset.score_answer(None, entry) == 0.0


def test_emoji_mystery_curriculum():
    """Test the emoji mystery curriculum functionality"""
    curriculum = EmojiMysteryCurriculum()

    base_value = {"size": 150, "seed": 1}

    # Test base configuration
    context = DefaultCurriculumContext(mode=RangeAttributeMode.UPPER_BOUND)
    base_cfg: EmojiMysteryConfig = curriculum.generate_configuration(base_value, context=context)
    assert base_cfg.seed == 1
    assert base_cfg.size == 150
    assert base_cfg.min_words_in_sentence == 5
    assert base_cfg.max_words_in_sentence == 10

    # Test incrementing attribute level
    curriculum.increment_attr_level("num_words_in_sentence")
    increased_cfg = curriculum.generate_configuration(base_value, context=context)
    assert increased_cfg.min_words_in_sentence == 10
    assert increased_cfg.max_words_in_sentence == 20

    # Test incrementing attribute level again
    curriculum.increment_attr_level("num_words_in_sentence")
    double_increased_cfg = curriculum.generate_configuration(base_value, context=context)
    assert double_increased_cfg.min_words_in_sentence == 20
    assert double_increased_cfg.max_words_in_sentence == 30

    # Test decrementing attribute level
    curriculum.decrement_attr_level("num_words_in_sentence")
    decreased_cfg = curriculum.generate_configuration(base_value, context=context)
    assert decreased_cfg.min_words_in_sentence == 10
    assert decreased_cfg.max_words_in_sentence == 20
