import pytest

from reasoning_gym.algorithmic.word_sequence_reversal import WordSequenceReversalConfig, WordSequenceReversalDataset


def test_word_sequence_reversal_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = WordSequenceReversalConfig(min_words=0)
        config.validate()

    with pytest.raises(AssertionError):
        config = WordSequenceReversalConfig(min_words=10, max_words=5)
        config.validate()


def test_word_sequence_reversal_dataset_deterministic():
    """Test that dataset generates same items with same seed"""
    config = WordSequenceReversalConfig(seed=42, size=10)
    dataset1 = WordSequenceReversalDataset(config)
    dataset2 = WordSequenceReversalDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_word_sequence_reversal_dataset_items():
    """Test basic properties of generated items"""
    config = WordSequenceReversalConfig(min_words=3, max_words=6, size=10, seed=42)
    dataset = WordSequenceReversalDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        # Check item structure
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata
        assert "num_words" in item["metadata"]
        assert "words" in item["metadata"]

        # Verify word count constraints
        words = item["metadata"]["words"]
        assert len(words) >= config.min_words
        assert len(words) <= config.max_words

        # Verify reversal is correct
        question_words = [w.strip() for w in item["question"].split(":")[1].strip().split(",")]
        answer_words = item["answer"].split(", ")
        assert answer_words == list(reversed(question_words))


def test_word_sequence_reversal_dataset_iteration():
    """Test that iteration respects dataset size"""
    config = WordSequenceReversalConfig(size=5, seed=42)
    dataset = WordSequenceReversalDataset(config)

    items = list(dataset)
    assert len(items) == config.size

    # Test multiple iterations yield same items
    assert items == list(dataset)


def test_word_sequence_reversal_text_preprocessing():
    """Test that text preprocessing handles edge cases"""
    config = WordSequenceReversalConfig(size=1, seed=42)
    dataset = WordSequenceReversalDataset(config)

    # Verify words were extracted from text
    assert len(dataset.words) > 0
    # Verify words contain only alphanumeric characters
    assert all(word.isalnum() for word in dataset.words)
