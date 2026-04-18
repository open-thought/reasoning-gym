import pytest

from reasoning_gym.combinatorics.combinatorics import CombinatoricsConfig, CombinatoricsCurriculum, CombinatoricsDataset


def test_config_validation():
    with pytest.raises(AssertionError):
        config = CombinatoricsConfig(min_n=1)
        config.validate()

    with pytest.raises(AssertionError):
        config = CombinatoricsConfig(min_n=10, max_n=5)
        config.validate()


def test_deterministic():
    config = CombinatoricsConfig(seed=42, size=10)
    ds1 = CombinatoricsDataset(config)
    ds2 = CombinatoricsDataset(config)
    for i in range(len(ds1)):
        assert ds1[i] == ds2[i]


def test_item_structure():
    config = CombinatoricsConfig(seed=42, size=50)
    ds = CombinatoricsDataset(config)
    for i in range(len(ds)):
        item = ds[i]
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item
        assert item["metadata"]["source_dataset"] == "combinatorics"


def test_answer_correctness():
    config = CombinatoricsConfig(seed=42, size=100)
    ds = CombinatoricsDataset(config)
    for i in range(len(ds)):
        item = ds[i]
        answer = item["answer"]
        assert answer.lstrip("-").isdigit(), f"Item {i}: answer '{answer}' is not an integer"


def test_score_wrong_answer():
    config = CombinatoricsConfig(seed=42, size=10)
    ds = CombinatoricsDataset(config)
    item = ds[0]
    score = ds.score_answer(item["answer"], item)
    assert score == 1.0


def test_curriculum():
    curriculum = CombinatoricsCurriculum()
    base_value = {"size": 50, "seed": 1}
    base_cfg = curriculum.generate_configuration(base_value)
    assert base_cfg.seed == 1

    curriculum.increment_attr_level("n_range")
    increased_cfg = curriculum.generate_configuration(base_value)
    assert increased_cfg.max_n >= base_cfg.max_n


def test_task_types():
    for task_type in ("ncr", "npr", "permutations_repetition", "inclusion_exclusion", "stars_and_bars", "pigeonhole"):
        config = CombinatoricsConfig(seed=42, size=10, task_types=(task_type,), task_weights=[1.0])
        ds = CombinatoricsDataset(config)
        for i in range(len(ds)):
            item = ds[i]
            assert item["metadata"]["task_type"] == task_type
