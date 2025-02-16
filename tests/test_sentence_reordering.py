import pytest

from reasoning_gym.algorithmic.sentence_reordering import SentenceReorderingConfig, SentenceReorderingDataset


@pytest.fixture
def config():
    return SentenceReorderingConfig(min_words_in_sentence=5, max_words_in_sentence=5, seed=42, size=10)


@pytest.fixture
def dataset(config):
    return SentenceReorderingDataset(config=config)


def test_config_validation(config):
    # Test that the config validation does not raise any exceptions
    try:
        config.validate()
    except Exception as e:
        pytest.fail(f"Config validation raised an exception: {e}")


def test_generate_sentence_dataset(dataset):
    sentence = "This is a test sentence for reordering"
    result = dataset._generate_sentence_dataset(sentence, seed=42, idx=0, shuffle=True)
    assert "input" in result
    assert "goal" in result
    assert result["input"] != result["goal"]
    assert sorted(result["input"].split()) == sorted(result["goal"].split())


def test_getitem(dataset, config):
    item = dataset[0]
    assert "question" in item
    assert "answer" in item
    assert "metadata" in item
    assert item["metadata"]["word_count"] >= config.min_words_in_sentence
    assert item["metadata"]["word_count"] <= config.max_words_in_sentence
    assert len(item["answer"].split()) == item["metadata"]["word_count"]


def test_key_error_in_getitem(dataset):
    # Modify the dataset to include an incorrect key
    def mock_generate_sentence_dataset(*args, **kwargs):
        return {"input": "mock input", "goal": "mock goal", "extra": "extra key"}

    dataset._generate_sentence_dataset = mock_generate_sentence_dataset

    with pytest.raises(KeyError):
        dataset[0]
