import pytest

from reasoning_gym.statistics.descriptive_stats import (
    DescriptiveStatsConfig,
    DescriptiveStatsCurriculum,
    DescriptiveStatsDataset,
)


def test_config_validation():
    with pytest.raises(AssertionError):
        config = DescriptiveStatsConfig(min_data_size=2)
        config.validate()

    with pytest.raises(AssertionError):
        config = DescriptiveStatsConfig(min_data_size=10, max_data_size=5)
        config.validate()


def test_deterministic():
    config = DescriptiveStatsConfig(seed=42, size=10)
    ds1 = DescriptiveStatsDataset(config)
    ds2 = DescriptiveStatsDataset(config)
    for i in range(len(ds1)):
        assert ds1[i] == ds2[i]


def test_item_structure():
    config = DescriptiveStatsConfig(seed=42, size=50)
    ds = DescriptiveStatsDataset(config)
    for i in range(len(ds)):
        item = ds[i]
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item
        assert item["metadata"]["source_dataset"] == "descriptive_stats"


def test_answer_correctness():
    config = DescriptiveStatsConfig(seed=42, size=50)
    ds = DescriptiveStatsDataset(config)
    for i in range(len(ds)):
        item = ds[i]
        score = ds.score_answer(item["answer"], item)
        assert score >= 1.0, f"Item {i}: oracle answer scored {score}"


def test_score_wrong_answer():
    config = DescriptiveStatsConfig(seed=42, size=10)
    ds = DescriptiveStatsDataset(config)
    item = ds[0]
    assert ds.score_answer(None, item) == 0.0
    assert ds.score_answer("definitely wrong", item) == 0.0


def test_curriculum():
    curriculum = DescriptiveStatsCurriculum()
    base_value = {"size": 50, "seed": 1}
    base_cfg = curriculum.generate_configuration(base_value)
    assert base_cfg.seed == 1

    curriculum.increment_attr_level("data_size")
    increased_cfg = curriculum.generate_configuration(base_value)
    assert increased_cfg.max_data_size >= base_cfg.max_data_size


def test_task_types():
    for task_type in ("mean", "median", "mode", "weighted_mean", "std_dev", "percentile", "z_score"):
        config = DescriptiveStatsConfig(seed=42, size=10, task_types=(task_type,), task_weights=[1.0])
        ds = DescriptiveStatsDataset(config)
        for i in range(len(ds)):
            item = ds[i]
            assert item["metadata"]["task_type"] == task_type
            score = ds.score_answer(item["answer"], item)
            assert score >= 1.0, f"Task {task_type}, item {i}: oracle scored {score}"
