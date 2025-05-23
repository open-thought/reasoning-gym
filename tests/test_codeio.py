import pytest

from reasoning_gym.code.codeio import CodeIOConfig, CodeIOCurriculum, CodeIODataset


def test_codeio_dataset():
    # Create a small CodeI/O reasoning dataset
    config = CodeIOConfig(size=10, seed=42)
    dataset = CodeIODataset(config)

    for i in range(10):
        item = dataset[i]

        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        assert "input_data" in item["metadata"]
        assert "output_data" in item["metadata"]

        # Score some correct and incorrect answers
        score = dataset.score_answer(answer=item["answer"], entry=item)
        assert score == 1.0
        # Incorrect answer (None)
        score = dataset.score_answer(answer=None, entry=item)
        assert score == 0.00
        # Incorrect answer (empty dict)
        score = dataset.score_answer(answer="{}", entry=item)
        assert score == 0.01


def test_codeio_config():
    # Test constraints on input probability
    with pytest.raises(AssertionError):
        CodeIOConfig(size=10, seed=42, input_prediction_probability=1.1).validate()

    with pytest.raises(AssertionError):
        CodeIOConfig(size=10, seed=42, input_prediction_probability=-0.1).validate()

    CodeIOConfig(size=10, seed=42, input_prediction_probability=0.1).validate()
    CodeIOConfig(size=10, seed=42, input_prediction_probability=0.9).validate()


def test_codeio_curriculum():
    curriculum = CodeIOCurriculum()

    base_value = {"size": 150, "seed": 1}

    base_cfg: CodeIOConfig = curriculum.generate_configuration(base_value)
    assert base_cfg.seed == 1
    assert base_cfg.size == 150
    assert base_cfg.difficulty == 6

    # test incrementing attribute level
    curriculum.increment_attr_level("difficulty")
    increased_cfg = curriculum.generate_configuration(base_value)
    assert increased_cfg.difficulty == 7

    # test decrementing attribute level
    curriculum.decrement_attr_level("difficulty")
    partially_decreased_cfg = curriculum.generate_configuration(base_value)
    assert partially_decreased_cfg.difficulty == 6
