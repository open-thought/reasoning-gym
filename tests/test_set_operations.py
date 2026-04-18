import pytest

from reasoning_gym.logic.set_operations import (
    SetOperationsConfig,
    SetOperationsCurriculum,
    SetOperationsDataset,
)


def test_config_validation():
    with pytest.raises(AssertionError):
        config = SetOperationsConfig(min_set_size=0)
        config.validate()

    with pytest.raises(AssertionError):
        config = SetOperationsConfig(min_set_size=10, max_set_size=5)
        config.validate()


def test_deterministic():
    config = SetOperationsConfig(seed=42, size=10)
    ds1 = SetOperationsDataset(config)
    ds2 = SetOperationsDataset(config)
    for i in range(len(ds1)):
        assert ds1[i] == ds2[i]


def test_item_structure():
    config = SetOperationsConfig(seed=42, size=50)
    ds = SetOperationsDataset(config)
    for i in range(len(ds)):
        item = ds[i]
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item
        assert item["metadata"]["source_dataset"] == "set_operations"


def test_answer_correctness():
    config = SetOperationsConfig(seed=42, size=50)
    ds = SetOperationsDataset(config)
    for i in range(len(ds)):
        item = ds[i]
        score = ds.score_answer(item["answer"], item)
        assert score >= 1.0, f"Item {i}: oracle answer scored {score}"


def test_score_wrong_answer():
    config = SetOperationsConfig(seed=42, size=10)
    ds = SetOperationsDataset(config)
    item = ds[0]
    assert ds.score_answer(None, item) == 0.0
    assert ds.score_answer("{999, 998, 997}", item) == 0.0


def test_curriculum():
    curriculum = SetOperationsCurriculum()
    base_value = {"size": 50, "seed": 1}
    base_cfg = curriculum.generate_configuration(base_value)
    assert base_cfg.seed == 1

    curriculum.increment_attr_level("set_size")
    increased_cfg = curriculum.generate_configuration(base_value)
    assert increased_cfg.max_set_size >= base_cfg.max_set_size


def test_task_types():
    for task_type in ("union", "intersection", "difference", "symmetric_difference", "cardinality", "power_set_size", "complement", "chained"):
        config = SetOperationsConfig(
            seed=42, size=10, task_types=(task_type,), task_weights=[1.0]
        )
        ds = SetOperationsDataset(config)
        for i in range(len(ds)):
            item = ds[i]
            assert item["metadata"]["task_type"] == task_type
            score = ds.score_answer(item["answer"], item)
            assert score >= 1.0, f"Task {task_type}, item {i}: oracle scored {score}"
