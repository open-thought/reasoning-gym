import pytest

from reasoning_gym.logic.self_reference import SelfReferenceConfig, SelfReferenceCurriculum, SelfReferenceDataset


def test_self_reference():
    """Test basic properties and solution of generated items"""

    # Easy
    config = SelfReferenceConfig(seed=42, size=20, difficulty=1)
    dataset = SelfReferenceDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Test the scoring
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0
        assert dataset.score_answer(answer=99, entry=item) == 0.0
        assert dataset.score_answer(answer="99", entry=item) == 0.0
        assert dataset.score_answer(answer=None, entry=item) == 0.0

    # # Medium
    config = SelfReferenceConfig(seed=42, size=1, difficulty=5)
    dataset = SelfReferenceDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Test the scoring
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0
        assert dataset.score_answer(answer=99, entry=item) == 0.0
        assert dataset.score_answer(answer="99", entry=item) == 0.0
        assert dataset.score_answer(answer=None, entry=item) == 0.0

    # # Hard
    config = SelfReferenceConfig(seed=42, size=1, difficulty=10)
    dataset = SelfReferenceDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Test the scoring
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0
        assert dataset.score_answer(answer=99, entry=item) == 0.0
        assert dataset.score_answer(answer="99", entry=item) == 0.0
        assert dataset.score_answer(answer=None, entry=item) == 0.0


def test_self_reference_curriculum():
    """Test the SelfReferenceCurriculum functionality"""

    curriculum = SelfReferenceCurriculum()

    base_value = {"size": 150, "seed": 1}

    # Test initial configuration
    base_cfg = curriculum.generate_configuration(base_value)
    assert base_cfg.seed == 1
    assert base_cfg.size == 150
    assert base_cfg.difficulty == 1  # Default level 0 maps to difficulty=1

    # Test incrementing difficulty attribute
    curriculum.increment_attr_level("difficulty")
    increased_cfg = curriculum.generate_configuration(base_value)
    assert increased_cfg.difficulty == 2
    assert increased_cfg.seed == 1  # Unchanged
    assert increased_cfg.size == 150  # Unchanged

    # Test incrementing difficulty attribute again
    curriculum.increment_attr_level("difficulty")
    increased_cfg_2 = curriculum.generate_configuration(base_value)
    assert increased_cfg_2.difficulty == 3

    # Test decrementing difficulty attribute
    curriculum.decrement_attr_level("difficulty")
    decreased_cfg = curriculum.generate_configuration(base_value)
    assert decreased_cfg.difficulty == 2

    # Test global level adjustments
    curriculum = SelfReferenceCurriculum()  # Reset curriculum
    assert curriculum.get_attr_level("difficulty") == 0  # Default level is 0, maps to difficulty=1

    # Increase global level
    curriculum.increment_global_level()
    assert curriculum.get_attr_level("difficulty") == 1

    global_level_cfg = curriculum.generate_configuration(base_value)
    assert global_level_cfg.difficulty == 2

    # Increase global level again
    curriculum.increment_global_level()
    assert curriculum.get_attr_level("difficulty") == 2

    global_level_cfg_2 = curriculum.generate_configuration(base_value)
    assert global_level_cfg_2.difficulty == 3

    # Decrease global level
    curriculum.decrement_global_level()
    assert curriculum.get_attr_level("difficulty") == 1

    global_level_cfg_3 = curriculum.generate_configuration(base_value)
    assert global_level_cfg_3.difficulty == 2

    # Test upper bound
    curriculum = SelfReferenceCurriculum()  # Reset curriculum
    for _ in range(15):  # Try going beyond max level (10)
        curriculum.increment_attr_level("difficulty")

    max_cfg = curriculum.generate_configuration(base_value)
    assert max_cfg.difficulty == 10  # Should be capped at 10 (the highest level)

    # Test lower bound
    curriculum = SelfReferenceCurriculum()  # Reset curriculum
    curriculum.decrement_attr_level("difficulty")  # Try going below min level

    min_cfg = curriculum.generate_configuration(base_value)
    assert min_cfg.difficulty == 1  # Should be capped at 1 (the lowest level)
