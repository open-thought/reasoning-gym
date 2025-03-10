import pytest

from reasoning_gym.games.sokoban import SokobanConfig, SokobanCurriculum, SokobanDataset


def test_sokoban():
    """Test basic properties and solution of generated items"""

    dataset = SokobanDataset(SokobanConfig(size=10, seed=1234))
    for i, item in enumerate(dataset):
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0

    # Easy
    config = SokobanConfig(seed=42, size=20)
    dataset = SokobanDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Test the scoring
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0
        assert dataset.score_answer(answer="RU", entry=item) == 0.0
        assert dataset.score_answer(answer=None, entry=item) == 0.0

    # Hard
    config = SokobanConfig(
        seed=42, min_h=15, max_h=20, min_w=15, max_w=20, min_boxes=10, max_boxes=15, size=3, max_depth=90
    )
    dataset = SokobanDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Test the scoring
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0
        assert dataset.score_answer(answer=None, entry=item) == 0.0

    # min == max ranges
    config = SokobanConfig(
        seed=42, min_h=11, max_h=11, min_w=11, max_w=11, min_boxes=11, max_boxes=11, size=3, max_depth=60
    )
    dataset = SokobanDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Test the scoring
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0
        assert dataset.score_answer(answer=None, entry=item) == 0.0


def test_sokoban_curriculum():
    """Test the SokobanCurriculum functionality"""
    curriculum = SokobanCurriculum()

    base_value = {"size": 150, "seed": 1}

    # Test initial configuration
    base_cfg = curriculum.generate_configuration(base_value)
    assert base_cfg.seed == 1
    assert base_cfg.size == 150
    assert base_cfg.min_w == 6 and base_cfg.max_w == 6
    assert base_cfg.min_h == 6 and base_cfg.max_h == 6
    assert base_cfg.min_boxes == 4  # Default value from SokobanConfig
    assert base_cfg.max_boxes == 10  # Default value from SokobanConfig

    # Test incrementing width attribute
    curriculum.increment_attr_level("width")
    width_cfg = curriculum.generate_configuration(base_value)
    assert width_cfg.min_w == 6 and width_cfg.max_w == 7
    assert width_cfg.min_h == 6 and width_cfg.max_h == 6  # Height unchanged

    # Test incrementing height attribute
    curriculum.increment_attr_level("height")
    both_cfg = curriculum.generate_configuration(base_value)
    assert both_cfg.min_w == 6 and both_cfg.max_w == 7  # Width preserved
    assert both_cfg.min_h == 6 and both_cfg.max_h == 7  # Height increased

    # Test decrementing width attribute
    curriculum.decrement_attr_level("width")
    height_only_cfg = curriculum.generate_configuration(base_value)
    assert height_only_cfg.min_w == 6 and height_only_cfg.max_w == 6  # Width reset
    assert height_only_cfg.min_h == 6 and height_only_cfg.max_h == 7  # Height preserved

    # Test global level adjustments
    curriculum = SokobanCurriculum()  # Reset curriculum
    assert curriculum.get_attr_level("width") == 0
    assert curriculum.get_attr_level("height") == 0

    # Increase global level
    curriculum.increment_global_level()
    assert curriculum.get_attr_level("width") == 1
    assert curriculum.get_attr_level("height") == 1

    global_level_cfg = curriculum.generate_configuration(base_value)
    assert global_level_cfg.min_w == 6 and global_level_cfg.max_w == 7
    assert global_level_cfg.min_h == 6 and global_level_cfg.max_h == 7

    # Increase global level again
    curriculum.increment_global_level()
    assert curriculum.get_attr_level("width") == 2
    assert curriculum.get_attr_level("height") == 2

    global_level_cfg_2 = curriculum.generate_configuration(base_value)
    assert global_level_cfg_2.min_w == 6 and global_level_cfg_2.max_w == 8
    assert global_level_cfg_2.min_h == 6 and global_level_cfg_2.max_h == 8

    # Decrease global level
    curriculum.decrement_global_level()
    assert curriculum.get_attr_level("width") == 1
    assert curriculum.get_attr_level("height") == 1

    global_level_cfg_3 = curriculum.generate_configuration(base_value)
    assert global_level_cfg_3.min_w == 6 and global_level_cfg_3.max_w == 7
    assert global_level_cfg_3.min_h == 6 and global_level_cfg_3.max_h == 7

    # Test upper bound
    curriculum = SokobanCurriculum()  # Reset curriculum
    for _ in range(10):  # Try going beyond max level
        curriculum.increment_attr_level("width")
        curriculum.increment_attr_level("height")

    max_cfg = curriculum.generate_configuration(base_value)
    assert max_cfg.min_w == 6 and max_cfg.max_w == 10  # Width capped at 10
    assert max_cfg.min_h == 6 and max_cfg.max_h == 10  # Height capped at 10

    # Test lower bound
    curriculum = SokobanCurriculum()  # Reset curriculum
    curriculum.decrement_attr_level("width")  # Try going below min level
    curriculum.decrement_attr_level("height")  # Try going below min level

    min_cfg = curriculum.generate_configuration(base_value)
    assert min_cfg.min_w == 6 and min_cfg.max_w == 6  # Width stays at min
    assert min_cfg.min_h == 6 and min_cfg.max_h == 6  # Height stays at min
