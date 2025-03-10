import pytest

from reasoning_gym.algorithmic.sentence_reordering import (
    SentenceReorderingConfig,
    SentenceReorderingCurriculum,
    SentenceReorderingDataset,
)


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


def test_sentence_reordering_curriculum():
    curriculum = SentenceReorderingCurriculum()

    base_value = {"size": 150, "seed": 1}

    base_cfg: SentenceReorderingConfig = curriculum.generate_configuration(base_value)
    assert base_cfg.seed == 1
    assert base_cfg.size == 150
    assert base_cfg.min_words_in_sentence == 5 and base_cfg.max_words_in_sentence == 20

    # test incrementing attribute levels
    curriculum.increment_attr_level("words_in_sentence")
    increased_cfg = curriculum.generate_configuration(base_value)
    assert increased_cfg.min_words_in_sentence == 5 and increased_cfg.max_words_in_sentence == 50

    # test decrementing attribute level
    curriculum.decrement_attr_level("words_in_sentence")
    partially_decreased_cfg = curriculum.generate_configuration(base_value)
    assert partially_decreased_cfg.min_words_in_sentence == 5 and partially_decreased_cfg.max_words_in_sentence == 20
