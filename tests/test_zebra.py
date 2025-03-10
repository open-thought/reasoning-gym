import pytest

from reasoning_gym.logic.zebra_puzzles import ZebraConfig, ZebraCurriculum, ZebraDataset


def test_zebra_deterministic():
    """Test that dataset generates same items with same seed"""
    config = ZebraConfig(seed=42, size=10, num_people=4, num_characteristics=4)
    dataset1 = ZebraDataset(config)
    dataset2 = ZebraDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_zebra_puzzles():
    """Test basic properties and solution of generated items"""
    config = ZebraConfig(seed=42, size=10, num_people=4, num_characteristics=4)
    dataset = ZebraDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Test the scoring
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0
        assert dataset.score_answer(answer=None, entry=item) == 0.0


def test_zebra_curriculum():
    """Test the ZebraCurriculum functionality"""

    curriculum = ZebraCurriculum()

    base_value = {"size": 150, "seed": 1}

    # Test initial configuration
    base_cfg = curriculum.generate_configuration(base_value)
    assert base_cfg.seed == 1
    assert base_cfg.size == 150
    assert base_cfg.num_people == 2  # Default level 0 maps to 2 people
    assert base_cfg.num_characteristics == 2  # Default level 0 maps to 2 characteristics

    # Test incrementing num_people attribute
    curriculum.increment_attr_level("num_people")
    people_cfg = curriculum.generate_configuration(base_value)
    assert people_cfg.num_people == 3  # Level 1 maps to 3 people
    assert people_cfg.num_characteristics == 2  # Unchanged

    # Test incrementing num_characteristics attribute
    curriculum.increment_attr_level("num_characteristics")
    both_cfg = curriculum.generate_configuration(base_value)
    assert both_cfg.num_people == 3  # Preserved
    assert both_cfg.num_characteristics == 3  # Level 1 maps to 3 characteristics

    # Test decrementing num_people attribute
    curriculum.decrement_attr_level("num_people")
    char_only_cfg = curriculum.generate_configuration(base_value)
    assert char_only_cfg.num_people == 2  # Back to level 0
    assert char_only_cfg.num_characteristics == 3  # Preserved

    # Test global level adjustments
    curriculum = ZebraCurriculum()  # Reset curriculum
    assert curriculum.get_attr_level("num_people") == 0
    assert curriculum.get_attr_level("num_characteristics") == 0

    # Increase global level
    curriculum.increment_global_level()
    assert curriculum.get_attr_level("num_people") == 1
    assert curriculum.get_attr_level("num_characteristics") == 1

    global_level_cfg = curriculum.generate_configuration(base_value)
    assert global_level_cfg.num_people == 3
    assert global_level_cfg.num_characteristics == 3

    # Increase global level again
    curriculum.increment_global_level()
    assert curriculum.get_attr_level("num_people") == 2
    assert curriculum.get_attr_level("num_characteristics") == 2

    global_level_cfg_2 = curriculum.generate_configuration(base_value)
    assert global_level_cfg_2.num_people == 4
    assert global_level_cfg_2.num_characteristics == 4

    # Decrease global level
    curriculum.decrement_global_level()
    assert curriculum.get_attr_level("num_people") == 1
    assert curriculum.get_attr_level("num_characteristics") == 1

    global_level_cfg_3 = curriculum.generate_configuration(base_value)
    assert global_level_cfg_3.num_people == 3
    assert global_level_cfg_3.num_characteristics == 3

    # Test upper bound
    curriculum = ZebraCurriculum()  # Reset curriculum
    for _ in range(10):  # Try going beyond max level
        curriculum.increment_attr_level("num_people")
        curriculum.increment_attr_level("num_characteristics")

    max_cfg = curriculum.generate_configuration(base_value)
    assert max_cfg.num_people == 7  # Capped at 7
    assert max_cfg.num_characteristics == 7  # Capped at 7

    # Test lower bound
    curriculum = ZebraCurriculum()  # Reset curriculum
    curriculum.decrement_attr_level("num_people")  # Try going below min level
    curriculum.decrement_attr_level("num_characteristics")  # Try going below min level

    min_cfg = curriculum.generate_configuration(base_value)
    assert min_cfg.num_people == 2  # Stays at 2
    assert min_cfg.num_characteristics == 2  # Stays at 2
