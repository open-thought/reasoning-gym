import pytest

from reasoning_gym.optimization.dynamic_programming import (
    DynamicProgrammingConfig,
    DynamicProgrammingCurriculum,
    DynamicProgrammingDataset,
)


def test_config_validation():
    with pytest.raises(AssertionError):
        config = DynamicProgrammingConfig(min_str_len=1)
        config.validate()

    with pytest.raises(AssertionError):
        config = DynamicProgrammingConfig(min_arr_len=2)
        config.validate()


def test_deterministic():
    config = DynamicProgrammingConfig(seed=42, size=10)
    ds1 = DynamicProgrammingDataset(config)
    ds2 = DynamicProgrammingDataset(config)
    for i in range(len(ds1)):
        assert ds1[i] == ds2[i]


def test_item_structure():
    config = DynamicProgrammingConfig(seed=42, size=50)
    ds = DynamicProgrammingDataset(config)
    for i in range(len(ds)):
        item = ds[i]
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item
        assert item["metadata"]["source_dataset"] == "dynamic_programming"


def test_answer_correctness():
    config = DynamicProgrammingConfig(seed=42, size=50)
    ds = DynamicProgrammingDataset(config)
    for i in range(len(ds)):
        item = ds[i]
        answer = item["answer"]
        assert answer.lstrip("-").isdigit(), f"Item {i}: answer '{answer}' is not an integer"


def test_score_answer():
    config = DynamicProgrammingConfig(seed=42, size=50)
    ds = DynamicProgrammingDataset(config)
    for i in range(len(ds)):
        item = ds[i]
        score = ds.score_answer(item["answer"], item)
        assert score == 1.0


def test_curriculum():
    curriculum = DynamicProgrammingCurriculum()
    base_value = {"size": 50, "seed": 1}
    base_cfg = curriculum.generate_configuration(base_value)
    assert base_cfg.seed == 1

    curriculum.increment_attr_level("max_str_len")
    increased_cfg = curriculum.generate_configuration(base_value)
    assert increased_cfg.max_str_len >= base_cfg.max_str_len


def test_task_types():
    for task_type in ("lcs", "coin_change", "lis", "edit_distance", "staircase"):
        config = DynamicProgrammingConfig(
            seed=42, size=10, task_types=(task_type,), task_weights=[1.0]
        )
        ds = DynamicProgrammingDataset(config)
        for i in range(len(ds)):
            item = ds[i]
            assert item["metadata"]["task_type"] == task_type
