import pytest

from reasoning_gym.optimization.linear_programming import (
    LinearProgrammingConfig,
    LinearProgrammingCurriculum,
    LinearProgrammingDataset,
)


def test_config_validation():
    with pytest.raises(AssertionError):
        config = LinearProgrammingConfig(min_coeff=0)
        config.validate()

    with pytest.raises(AssertionError):
        config = LinearProgrammingConfig(num_constraints=1)
        config.validate()


def test_deterministic():
    config = LinearProgrammingConfig(seed=42, size=10)
    ds1 = LinearProgrammingDataset(config)
    ds2 = LinearProgrammingDataset(config)
    for i in range(len(ds1)):
        assert ds1[i] == ds2[i]


def test_item_structure():
    config = LinearProgrammingConfig(seed=42, size=50)
    ds = LinearProgrammingDataset(config)
    for i in range(len(ds)):
        item = ds[i]
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item
        assert item["metadata"]["source_dataset"] == "linear_programming"


def test_answer_correctness():
    config = LinearProgrammingConfig(seed=42, size=50)
    ds = LinearProgrammingDataset(config)
    for i in range(len(ds)):
        item = ds[i]
        score = ds.score_answer(item["answer"], item)
        assert score >= 1.0, f"Item {i}: oracle answer scored {score}"


def test_score_wrong_answer():
    config = LinearProgrammingConfig(seed=42, size=10)
    ds = LinearProgrammingDataset(config)
    item = ds[0]
    assert ds.score_answer(None, item) == 0.0
    assert ds.score_answer("not a number", item) == 0.0


def test_curriculum():
    curriculum = LinearProgrammingCurriculum()
    base_value = {"size": 50, "seed": 1}
    base_cfg = curriculum.generate_configuration(base_value)
    assert base_cfg.seed == 1

    curriculum.increment_attr_level("num_constraints")
    increased_cfg = curriculum.generate_configuration(base_value)
    assert increased_cfg.num_constraints >= base_cfg.num_constraints
