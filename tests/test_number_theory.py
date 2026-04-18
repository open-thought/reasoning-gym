import pytest

from reasoning_gym.arithmetic.number_theory import (
    NumberTheoryConfig,
    NumberTheoryCurriculum,
    NumberTheoryDataset,
)


def test_config_validation():
    with pytest.raises(AssertionError):
        config = NumberTheoryConfig(min_value=1)
        config.validate()

    with pytest.raises(AssertionError):
        config = NumberTheoryConfig(min_value=50, max_value=10)
        config.validate()


def test_deterministic():
    config = NumberTheoryConfig(seed=42, size=10)
    ds1 = NumberTheoryDataset(config)
    ds2 = NumberTheoryDataset(config)
    for i in range(len(ds1)):
        assert ds1[i] == ds2[i]


def test_item_structure():
    config = NumberTheoryConfig(seed=42, size=50)
    ds = NumberTheoryDataset(config)
    for i in range(len(ds)):
        item = ds[i]
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item
        assert item["metadata"]["source_dataset"] == "number_theory"


def test_answer_correctness():
    config = NumberTheoryConfig(seed=42, size=50)
    ds = NumberTheoryDataset(config)
    for i in range(len(ds)):
        item = ds[i]
        score = ds.score_answer(item["answer"], item)
        assert score >= 1.0, f"Item {i}: oracle answer scored {score}"


def test_diophantine_verification():
    config = NumberTheoryConfig(seed=42, size=20, task_types=("diophantine",), task_weights=[1.0])
    ds = NumberTheoryDataset(config)
    for i in range(len(ds)):
        item = ds[i]
        a = item["metadata"]["a"]
        b = item["metadata"]["b"]
        c = item["metadata"]["c"]
        parts = {}
        for part in item["answer"].split(","):
            k, v = part.split("=")
            parts[k.strip()] = int(v.strip())
        assert a * parts["x"] + b * parts["y"] == c


def test_score_wrong_answer():
    config = NumberTheoryConfig(seed=42, size=10)
    ds = NumberTheoryDataset(config)
    item = ds[0]
    assert ds.score_answer(None, item) == 0.0


def test_curriculum():
    curriculum = NumberTheoryCurriculum()
    base_value = {"size": 50, "seed": 1}
    base_cfg = curriculum.generate_configuration(base_value)
    assert base_cfg.seed == 1

    curriculum.increment_attr_level("value_range")
    increased_cfg = curriculum.generate_configuration(base_value)
    assert increased_cfg.max_value >= base_cfg.max_value


def test_task_types():
    for task_type in ("mod_arith", "mod_exp", "totient", "crt", "mod_inverse", "diophantine"):
        config = NumberTheoryConfig(seed=42, size=10, task_types=(task_type,), task_weights=[1.0])
        ds = NumberTheoryDataset(config)
        for i in range(len(ds)):
            item = ds[i]
            assert item["metadata"]["task_type"] == task_type
            score = ds.score_answer(item["answer"], item)
            assert score >= 1.0, f"Task {task_type}, item {i}: oracle scored {score}"
