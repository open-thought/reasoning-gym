import pytest

from reasoning_gym.languages.regex_puzzles import (
    RegexPuzzlesConfig,
    RegexPuzzlesCurriculum,
    RegexPuzzlesDataset,
)


def test_config_validation():
    with pytest.raises(AssertionError):
        config = RegexPuzzlesConfig(min_dfa_states=1)
        config.validate()

    with pytest.raises(AssertionError):
        config = RegexPuzzlesConfig(min_dfa_states=10, max_dfa_states=5)
        config.validate()


def test_deterministic():
    config = RegexPuzzlesConfig(seed=42, size=10)
    ds1 = RegexPuzzlesDataset(config)
    ds2 = RegexPuzzlesDataset(config)
    for i in range(len(ds1)):
        assert ds1[i] == ds2[i]


def test_item_structure():
    config = RegexPuzzlesConfig(seed=42, size=50)
    ds = RegexPuzzlesDataset(config)
    for i in range(len(ds)):
        item = ds[i]
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item
        assert item["metadata"]["source_dataset"] == "regex_puzzles"


def test_answer_correctness():
    config = RegexPuzzlesConfig(seed=42, size=50)
    ds = RegexPuzzlesDataset(config)
    for i in range(len(ds)):
        item = ds[i]
        score = ds.score_answer(item["answer"], item)
        assert score >= 1.0, f"Item {i}: oracle answer scored {score}"


def test_score_wrong_answer():
    config = RegexPuzzlesConfig(seed=42, size=10)
    ds = RegexPuzzlesDataset(config)
    item = ds[0]
    assert ds.score_answer(None, item) == 0.0


def test_curriculum():
    curriculum = RegexPuzzlesCurriculum()
    base_value = {"size": 50, "seed": 1}
    base_cfg = curriculum.generate_configuration(base_value)
    assert base_cfg.seed == 1

    curriculum.increment_attr_level("max_dfa_states")
    increased_cfg = curriculum.generate_configuration(base_value)
    assert increased_cfg.max_dfa_states >= base_cfg.max_dfa_states


def test_task_types():
    for task_type in ("string_generation", "extraction", "dfa_state", "dfa_prefix"):
        config = RegexPuzzlesConfig(
            seed=42, size=10, task_types=(task_type,), task_weights=[1.0]
        )
        ds = RegexPuzzlesDataset(config)
        for i in range(len(ds)):
            item = ds[i]
            assert item["metadata"]["task_type"] == task_type
            score = ds.score_answer(item["answer"], item)
            assert score >= 1.0, f"Task {task_type}, item {i}: oracle scored {score}"
