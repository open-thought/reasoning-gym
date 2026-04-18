import pytest

from reasoning_gym.graphs.job_scheduling import (
    JobSchedulingConfig,
    JobSchedulingCurriculum,
    JobSchedulingDataset,
)


def test_config_validation():
    with pytest.raises(AssertionError):
        config = JobSchedulingConfig(min_jobs=2)
        config.validate()

    with pytest.raises(AssertionError):
        config = JobSchedulingConfig(min_jobs=10, max_jobs=5)
        config.validate()


def test_deterministic():
    config = JobSchedulingConfig(seed=42, size=10)
    ds1 = JobSchedulingDataset(config)
    ds2 = JobSchedulingDataset(config)
    for i in range(len(ds1)):
        assert ds1[i] == ds2[i]


def test_item_structure():
    config = JobSchedulingConfig(seed=42, size=50)
    ds = JobSchedulingDataset(config)
    for i in range(len(ds)):
        item = ds[i]
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item
        assert item["metadata"]["source_dataset"] == "job_scheduling"


def test_answer_correctness():
    config = JobSchedulingConfig(seed=42, size=50)
    ds = JobSchedulingDataset(config)
    for i in range(len(ds)):
        item = ds[i]
        score = ds.score_answer(item["answer"], item)
        assert score >= 1.0, f"Item {i}: oracle answer scored {score}"


def test_task_ordering_verification():
    config = JobSchedulingConfig(
        seed=42, size=20, task_types=("task_ordering",), task_weights=[1.0]
    )
    ds = JobSchedulingDataset(config)
    for i in range(len(ds)):
        item = ds[i]
        order = item["answer"].split(", ")
        deps = item["metadata"]["deps"]
        pos = {name: j for j, name in enumerate(order)}
        for j, prereqs in deps.items():
            for p in prereqs:
                assert pos[p] < pos[j], f"Dependency {p} -> {j} violated"


def test_score_wrong_answer():
    config = JobSchedulingConfig(seed=42, size=10)
    ds = JobSchedulingDataset(config)
    item = ds[0]
    assert ds.score_answer(None, item) == 0.0


def test_curriculum():
    curriculum = JobSchedulingCurriculum()
    base_value = {"size": 50, "seed": 1}
    base_cfg = curriculum.generate_configuration(base_value)
    assert base_cfg.seed == 1

    curriculum.increment_attr_level("max_jobs")
    increased_cfg = curriculum.generate_configuration(base_value)
    assert increased_cfg.max_jobs >= base_cfg.max_jobs


def test_task_types():
    for task_type in ("critical_path", "interval_scheduling", "task_ordering"):
        config = JobSchedulingConfig(
            seed=42, size=10, task_types=(task_type,), task_weights=[1.0]
        )
        ds = JobSchedulingDataset(config)
        for i in range(len(ds)):
            item = ds[i]
            assert item["metadata"]["task_type"] == task_type
            score = ds.score_answer(item["answer"], item)
            assert score >= 1.0, f"Task {task_type}, item {i}: oracle scored {score}"
