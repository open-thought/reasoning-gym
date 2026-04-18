import pytest

from reasoning_gym.algebra.complex_advanced import (
    ComplexAdvancedConfig,
    ComplexAdvancedCurriculum,
    ComplexAdvancedDataset,
)


def test_config_validation():
    with pytest.raises(AssertionError):
        config = ComplexAdvancedConfig(min_real=0)
        config.validate()

    with pytest.raises(AssertionError):
        config = ComplexAdvancedConfig(min_real=10, max_real=5)
        config.validate()

    with pytest.raises(AssertionError):
        config = ComplexAdvancedConfig(task_types=("invalid",))
        config.validate()


def test_deterministic():
    config = ComplexAdvancedConfig(seed=42, size=10)
    ds1 = ComplexAdvancedDataset(config)
    ds2 = ComplexAdvancedDataset(config)
    for i in range(len(ds1)):
        assert ds1[i] == ds2[i]


def test_item_structure():
    config = ComplexAdvancedConfig(seed=42, size=50)
    ds = ComplexAdvancedDataset(config)
    for i in range(len(ds)):
        item = ds[i]
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item
        assert item["metadata"]["source_dataset"] == "complex_advanced"


def test_answer_correctness():
    config = ComplexAdvancedConfig(seed=42, size=50)
    ds = ComplexAdvancedDataset(config)
    for i in range(len(ds)):
        item = ds[i]
        score = ds.score_answer(item["answer"], item)
        assert score >= 1.0, f"Item {i}: oracle answer scored {score}, expected 1.0"


def test_score_wrong_answer():
    config = ComplexAdvancedConfig(seed=42, size=10)
    ds = ComplexAdvancedDataset(config)
    item = ds[0]
    assert ds.score_answer(None, item) == 0.0
    assert ds.score_answer("completely wrong", item) == 0.0


def test_curriculum():
    curriculum = ComplexAdvancedCurriculum()
    base_value = {"size": 50, "seed": 1}
    base_cfg = curriculum.generate_configuration(base_value)
    assert base_cfg.seed == 1
    assert base_cfg.size == 50

    curriculum.increment_attr_level("max_real")
    increased_cfg = curriculum.generate_configuration(base_value)
    assert increased_cfg.max_real >= base_cfg.max_real

    curriculum.decrement_attr_level("max_real")
    restored_cfg = curriculum.generate_configuration(base_value)
    assert restored_cfg.max_real == base_cfg.max_real


def test_task_types():
    for task_type in ("polar", "euler", "inverse", "sqrt", "quadratic"):
        config = ComplexAdvancedConfig(seed=42, size=10, task_types=(task_type,), task_weights=[1.0])
        ds = ComplexAdvancedDataset(config)
        for i in range(len(ds)):
            item = ds[i]
            assert item["metadata"]["task_type"] == task_type
            score = ds.score_answer(item["answer"], item)
            assert score >= 1.0, f"Task {task_type}, item {i}: oracle scored {score}"
