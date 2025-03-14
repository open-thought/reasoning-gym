import json

import pytest

from reasoning_gym.algorithmic.jugs import JugsConfig, JugsCurriculum, JugsDataset


def test_jugs():
    """Test basic properties and solution of generated items"""
    config = JugsConfig(seed=42, size=1000, num_jugs=3, difficulty=5)
    dataset = JugsDataset(config)

    # easy
    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Test the scoring
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0
        assert dataset.score_answer(answer=None, entry=item) == 0.0

    config = JugsConfig(seed=42, size=1, num_jugs=3, difficulty=50)
    dataset = JugsDataset(config)

    # med
    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Test the scoring
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0
        assert dataset.score_answer(answer=None, entry=item) == 0.0

    config = JugsConfig(seed=42, size=1, num_jugs=3, difficulty=99)
    dataset = JugsDataset(config)

    # hard
    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item
        assert "difficulty" in item["metadata"]

        # Test the scoring
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0
        assert dataset.score_answer(answer=None, entry=item) == 0.0


def test_game_of_life_curriculum():
    """Test the curriculum for complex arithmetic."""
    curriculum = JugsCurriculum()
    base_value = {"size": 150, "seed": 1}

    base_cfg: JugsCurriculum = curriculum.generate_configuration(base_value)

    assert base_cfg.seed == 1
    assert base_cfg.size == 150
    assert base_cfg.num_jugs == 3
    assert base_cfg.difficulty == 2

    # Test and validate increase in levels
    curriculum.increment_attr_level("num_jugs")
    curriculum.increment_attr_level("difficulty")

    increased_cfg: JugsCurriculum = curriculum.generate_configuration(base_value)
    assert increased_cfg.num_jugs == 4
    assert increased_cfg.difficulty == 4

    # Test and validate decrease in levels
    curriculum.decrement_attr_level("num_jugs")
    curriculum.decrement_attr_level("difficulty")

    decreased_cfg: JugsCurriculum = curriculum.generate_configuration(base_value)
    assert decreased_cfg.num_jugs == 3
    assert decreased_cfg.difficulty == 2

    # Test upper bound boundary condition
    for _ in range(10):
        curriculum.increment_attr_level("num_jugs")
        curriculum.increment_attr_level("difficulty")
    upper_bound_cfg: JugsCurriculum = curriculum.generate_configuration(base_value)
    assert upper_bound_cfg.num_jugs == 7
    assert upper_bound_cfg.difficulty == 8

    # Test lower bound boundary condition
    for _ in range(10):
        curriculum.decrement_attr_level("num_jugs")
        curriculum.decrement_attr_level("difficulty")
    lower_bound_cfg: JugsCurriculum = curriculum.generate_configuration(base_value)
    assert lower_bound_cfg.num_jugs == 3
    assert lower_bound_cfg.difficulty == 2
