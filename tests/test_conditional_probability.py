import pytest

from reasoning_gym.probability.conditional_probability import (
    ConditionalProbabilityConfig,
    ConditionalProbabilityCurriculum,
    ConditionalProbabilityDataset,
)


def test_config_validation():
    with pytest.raises(AssertionError):
        config = ConditionalProbabilityConfig(min_total_items=1)
        config.validate()

    with pytest.raises(AssertionError):
        config = ConditionalProbabilityConfig(min_total_items=20, max_total_items=5)
        config.validate()


def test_deterministic():
    config = ConditionalProbabilityConfig(seed=42, size=10)
    ds1 = ConditionalProbabilityDataset(config)
    ds2 = ConditionalProbabilityDataset(config)
    for i in range(len(ds1)):
        assert ds1[i] == ds2[i]


def test_item_structure():
    config = ConditionalProbabilityConfig(seed=42, size=50)
    ds = ConditionalProbabilityDataset(config)
    for i in range(len(ds)):
        item = ds[i]
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item
        assert item["metadata"]["source_dataset"] == "conditional_probability"


def test_answer_correctness():
    config = ConditionalProbabilityConfig(seed=42, size=50)
    ds = ConditionalProbabilityDataset(config)
    for i in range(len(ds)):
        item = ds[i]
        score = ds.score_answer(item["answer"], item)
        assert score >= 1.0, f"Item {i}: oracle answer scored {score}"


def test_score_wrong_answer():
    config = ConditionalProbabilityConfig(seed=42, size=10)
    ds = ConditionalProbabilityDataset(config)
    item = ds[0]
    assert ds.score_answer(None, item) == 0.0
    assert ds.score_answer("not a fraction", item) == 0.0


def test_curriculum():
    curriculum = ConditionalProbabilityCurriculum()
    base_value = {"size": 50, "seed": 1}
    base_cfg = curriculum.generate_configuration(base_value)
    assert base_cfg.seed == 1
    assert base_cfg.size == 50

    curriculum.increment_attr_level("total_items")
    increased_cfg = curriculum.generate_configuration(base_value)
    assert increased_cfg.max_total_items >= base_cfg.max_total_items


def test_task_types():
    for task_type in ("bayes", "dependent_draws", "contingency_table"):
        config = ConditionalProbabilityConfig(
            seed=42, size=10, task_types=(task_type,), task_weights=[1.0]
        )
        ds = ConditionalProbabilityDataset(config)
        for i in range(len(ds)):
            item = ds[i]
            assert item["metadata"]["task_type"] == task_type
            score = ds.score_answer(item["answer"], item)
            assert score >= 1.0, f"Task {task_type}, item {i}: oracle scored {score}"
