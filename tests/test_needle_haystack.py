import pytest

from reasoning_gym.cognition.needle_haystack import (
    NeedleHaystackConfig,
    NeedleHaystackCurriculum,
    NeedleHaystackDataset,
)


def test_needle_haystack():
    """Test basic properties and solution of generated items"""
    config = NeedleHaystackConfig(seed=42, size=50, min_num_statements=50, max_num_statements=50)
    dataset = NeedleHaystackDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Test the scoring
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0
        assert dataset.score_answer(answer="david bowie rules", entry=item) == 0.0
        assert dataset.score_answer(answer=None, entry=item) == 0.0

    config = NeedleHaystackConfig(seed=42, size=1, min_num_statements=500, max_num_statements=500)
    dataset = NeedleHaystackDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Test the scoring
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0
        assert dataset.score_answer(answer=None, entry=item) == 0.0

    config = NeedleHaystackConfig(seed=42, size=1, min_num_statements=5000, max_num_statements=5000)
    dataset = NeedleHaystackDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Test the scoring
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0
        assert dataset.score_answer(answer=None, entry=item) == 0.0

    config = NeedleHaystackConfig(seed=42, size=1, min_num_statements=50000, max_num_statements=50000)
    dataset = NeedleHaystackDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Test the scoring
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0
        assert dataset.score_answer(answer=None, entry=item) == 0.0

    config = NeedleHaystackConfig(seed=42, size=1, min_num_statements=500000, max_num_statements=500000)
    dataset = NeedleHaystackDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Test the scoring
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0
        assert dataset.score_answer(answer=None, entry=item) == 0.0


def test_needle_haystack_curriculum():
    curriculum = NeedleHaystackCurriculum()

    base_value = {"size": 150, "seed": 1}

    base_cfg: NeedleHaystackConfig = curriculum.generate_configuration(base_value)
    assert base_cfg.seed == 1
    assert base_cfg.size == 150
    assert base_cfg.min_num_statements == 10 and base_cfg.max_num_statements == 100

    # test incrementing attribute levels
    curriculum.increment_attr_level("num_statements")
    increased_cfg = curriculum.generate_configuration(base_value)
    assert increased_cfg.min_num_statements == 10 and increased_cfg.max_num_statements == 500

    # test decrementing attribute level for num_statements again
    curriculum.decrement_attr_level("num_statements")
    partially_decreased_cfg = curriculum.generate_configuration(base_value)
    assert partially_decreased_cfg.min_num_statements == 10 and partially_decreased_cfg.max_num_statements == 100
