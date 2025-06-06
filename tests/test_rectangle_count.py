import pytest

from reasoning_gym.cognition.rectangle_count import (
    RectangleCountConfig,
    RectangleCountCurriculum,
    RectangleCountDataset,
)


def test_dice():
    """Test basic properties and solution of generated items"""
    config = RectangleCountConfig(seed=42, size=50, max_rectangles=15, width=40, height=40)
    dataset = RectangleCountDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Test the scoring
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0
        assert dataset.score_answer(answer=None, entry=item) == 0.0


def test_rc_curriculum():
    """Test the RectangleCountCurriculum functionality"""
    curriculum = RectangleCountCurriculum()

    base_value = {"size": 150, "seed": 1}

    # Test the initial configuration
    base_cfg: RectangleCountConfig = curriculum.generate_configuration(base_value)
    assert base_cfg.seed == 1
    assert base_cfg.size == 150
    assert base_cfg.max_rectangles == 5

    # Test incrementing the number_rectangles attribute
    curriculum.increment_attr_level("max_rectangles")
    increased_cfg = curriculum.generate_configuration(base_value)
    assert increased_cfg.max_rectangles == 10

    # Test another increment
    curriculum.increment_attr_level("max_rectangles")
    increased_cfg = curriculum.generate_configuration(base_value)
    assert increased_cfg.max_rectangles == 15

    # Test decrementing
    curriculum.decrement_attr_level("max_rectangles")
    decreased_cfg = curriculum.generate_configuration(base_value)
    assert decreased_cfg.max_rectangles == 10
