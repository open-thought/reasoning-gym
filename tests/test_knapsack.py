import pytest

from reasoning_gym.optimization.knapsack import (
    KnapsackConfig,
    KnapsackCurriculum,
    KnapsackDataset,
    _solve_knapsack,
)


def test_config_validation():
    with pytest.raises(AssertionError):
        config = KnapsackConfig(min_items=1)
        config.validate()

    with pytest.raises(AssertionError):
        config = KnapsackConfig(min_items=10, max_items=5)
        config.validate()


def test_deterministic():
    config = KnapsackConfig(seed=42, size=10)
    ds1 = KnapsackDataset(config)
    ds2 = KnapsackDataset(config)
    for i in range(len(ds1)):
        assert ds1[i] == ds2[i]


def test_item_structure():
    config = KnapsackConfig(seed=42, size=50)
    ds = KnapsackDataset(config)
    for i in range(len(ds)):
        item = ds[i]
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item
        assert item["metadata"]["source_dataset"] == "knapsack"


def test_answer_correctness():
    config = KnapsackConfig(seed=42, size=50)
    ds = KnapsackDataset(config)
    for i in range(len(ds)):
        item = ds[i]
        weights = item["metadata"]["weights"]
        values = item["metadata"]["values"]
        capacity = item["metadata"]["capacity"]
        expected = _solve_knapsack(weights, values, capacity)
        assert int(item["answer"]) == expected, f"Item {i}: answer mismatch"


def test_score_answer():
    config = KnapsackConfig(seed=42, size=10)
    ds = KnapsackDataset(config)
    for i in range(len(ds)):
        item = ds[i]
        score = ds.score_answer(item["answer"], item)
        assert score == 1.0


def test_curriculum():
    curriculum = KnapsackCurriculum()
    base_value = {"size": 50, "seed": 1}
    base_cfg = curriculum.generate_configuration(base_value)
    assert base_cfg.seed == 1

    curriculum.increment_attr_level("item_count")
    increased_cfg = curriculum.generate_configuration(base_value)
    assert increased_cfg.max_items >= base_cfg.max_items
