import pytest

from reasoning_gym.algebra.linear_algebra import (
    LinearAlgebraConfig,
    LinearAlgebraCurriculum,
    LinearAlgebraDataset,
)


def test_config_validation():
    with pytest.raises(AssertionError):
        config = LinearAlgebraConfig(min_dim=1)
        config.validate()

    with pytest.raises(AssertionError):
        config = LinearAlgebraConfig(max_dim=5)
        config.validate()


def test_deterministic():
    config = LinearAlgebraConfig(seed=42, size=10)
    ds1 = LinearAlgebraDataset(config)
    ds2 = LinearAlgebraDataset(config)
    for i in range(len(ds1)):
        assert ds1[i] == ds2[i]


def test_item_structure():
    config = LinearAlgebraConfig(seed=42, size=50)
    ds = LinearAlgebraDataset(config)
    for i in range(len(ds)):
        item = ds[i]
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item
        assert item["metadata"]["source_dataset"] == "linear_algebra"


def test_answer_correctness():
    config = LinearAlgebraConfig(seed=42, size=50)
    ds = LinearAlgebraDataset(config)
    for i in range(len(ds)):
        item = ds[i]
        score = ds.score_answer(item["answer"], item)
        assert score >= 1.0, f"Item {i}: oracle answer scored {score}"


def test_solve_system_verification():
    config = LinearAlgebraConfig(seed=42, size=20, task_types=("solve_system",), task_weights=[1.0])
    ds = LinearAlgebraDataset(config)
    for i in range(len(ds)):
        item = ds[i]
        score = ds.score_answer(item["answer"], item)
        assert score >= 1.0


def test_score_wrong_answer():
    config = LinearAlgebraConfig(seed=42, size=10)
    ds = LinearAlgebraDataset(config)
    item = ds[0]
    assert ds.score_answer(None, item) == 0.0
    assert ds.score_answer("totally wrong", item) == 0.0


def test_curriculum():
    curriculum = LinearAlgebraCurriculum()
    base_value = {"size": 50, "seed": 1}
    base_cfg = curriculum.generate_configuration(base_value)
    assert base_cfg.seed == 1

    curriculum.increment_attr_level("max_dim")
    increased_cfg = curriculum.generate_configuration(base_value)
    assert increased_cfg.max_dim >= base_cfg.max_dim


def test_task_types():
    for task_type in ("matrix_multiply", "determinant", "inverse", "solve_system", "eigenvalues"):
        config = LinearAlgebraConfig(seed=42, size=10, task_types=(task_type,), task_weights=[1.0])
        ds = LinearAlgebraDataset(config)
        for i in range(len(ds)):
            item = ds[i]
            assert item["metadata"]["task_type"] == task_type
            score = ds.score_answer(item["answer"], item)
            assert score >= 1.0, f"Task {task_type}, item {i}: oracle scored {score}"
