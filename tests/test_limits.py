import pytest

from reasoning_gym.algebra.limits import (
    LimitsConfig,
    LimitsCurriculum,
    LimitsDataset,
)


def test_config_validation():
    with pytest.raises(AssertionError):
        config = LimitsConfig(max_coeff=0)
        config.validate()

    with pytest.raises(AssertionError):
        config = LimitsConfig(max_degree=0)
        config.validate()


def test_deterministic():
    config = LimitsConfig(seed=42, size=10)
    ds1 = LimitsDataset(config)
    ds2 = LimitsDataset(config)
    for i in range(len(ds1)):
        assert ds1[i] == ds2[i]


def test_item_structure():
    config = LimitsConfig(seed=42, size=50)
    ds = LimitsDataset(config)
    for i in range(len(ds)):
        item = ds[i]
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item
        assert item["metadata"]["source_dataset"] == "limits"


def test_answer_correctness():
    config = LimitsConfig(seed=42, size=50)
    ds = LimitsDataset(config)
    for i in range(len(ds)):
        item = ds[i]
        score = ds.score_answer(item["answer"], item)
        assert score >= 1.0, f"Item {i}: oracle answer scored {score}"


def test_score_wrong_answer():
    config = LimitsConfig(seed=42, size=10)
    ds = LimitsDataset(config)
    item = ds[0]
    assert ds.score_answer(None, item) == 0.0
    assert ds.score_answer("wrong", item) == 0.0


def test_curriculum():
    curriculum = LimitsCurriculum()
    base_value = {"size": 50, "seed": 1}
    base_cfg = curriculum.generate_configuration(base_value)
    assert base_cfg.seed == 1

    curriculum.increment_attr_level("max_coeff")
    increased_cfg = curriculum.generate_configuration(base_value)
    assert increased_cfg.max_coeff >= base_cfg.max_coeff


def test_task_types():
    for task_type in ("polynomial_cancel", "rational_infinity", "direct_sub", "squeeze"):
        config = LimitsConfig(seed=42, size=10, task_types=(task_type,), task_weights=[1.0])
        ds = LimitsDataset(config)
        for i in range(len(ds)):
            item = ds[i]
            assert item["metadata"]["task_type"] == task_type
            score = ds.score_answer(item["answer"], item)
            assert score >= 1.0, f"Task {task_type}, item {i}: oracle scored {score}"
