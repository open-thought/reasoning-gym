"""Tests for Course Schedule puzzle generation"""

import pytest

from reasoning_gym.graphs.course_schedule import CourseScheduleConfig, CourseScheduleCurriculum, CourseScheduleDataset


def test_course_schedule_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = CourseScheduleConfig(min_num_courses=2)  # must be >= 3
        config.validate()

    with pytest.raises(AssertionError):
        config = CourseScheduleConfig(min_num_courses=6, max_num_courses=5)  # min > max
        config.validate()

    with pytest.raises(AssertionError):
        config = CourseScheduleConfig(min_num_prerequisites=-1)  # neg not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = CourseScheduleConfig(min_num_prerequisites=5, max_num_prerequisites=4)  # min > max
        config.validate()

    with pytest.raises(AssertionError):
        config = CourseScheduleConfig(max_num_prerequisites=0)  # Zero not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = CourseScheduleConfig(p_solvable=-0.1)  # < 0 not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = CourseScheduleConfig(p_solvable=1.1)  # > 1 not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = CourseScheduleConfig(p_solvable=1.1)  # > 1 not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = CourseScheduleConfig(min_cycle_length=2)  # < 3 not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = CourseScheduleConfig(min_cycle_length=3, max_cycle_length=2)  # min_cycle_length > max_cycle_length
        config.validate()


def test_course_schedule_dataset_deterministic():
    """Test that dataset generates same items with same seed"""
    config = CourseScheduleConfig(seed=42, size=10)
    dataset1 = CourseScheduleDataset(config)
    dataset2 = CourseScheduleDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_course_schedule_dataset_items():
    """Test basic properties of generated items"""
    config = CourseScheduleConfig(max_num_courses=15, size=10, seed=42)
    dataset = CourseScheduleDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        # Check item structure
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata
        assert "courses" in item["metadata"]
        assert "prerequisites" in item["metadata"]
        assert "solution" in item["metadata"]
        assert "solvable" in item["metadata"]

        courses = item["metadata"]["courses"]
        prerequisites = item["metadata"]["prerequisites"]
        solvable = item["metadata"]["solvable"]  # Solution dictated by p_solvable
        solution = item["metadata"]["solution"]  # Solution obtained from topological sort

        # Verify metadata
        assert len(courses) <= config.max_num_courses
        assert len(prerequisites) <= config.max_num_prerequisites * len(courses)
        assert all(len(prereq) == 2 for prereq in prerequisites)
        for course, prereq in prerequisites:
            assert course < len(courses)
            assert prereq < len(courses)
            assert course != prereq
        assert solution == solvable


def test_course_schedule_dataset_iteration():
    """Test that iteration respects dataset size"""
    config = CourseScheduleConfig(size=5, seed=42)
    dataset = CourseScheduleDataset(config)

    items = list(dataset)
    assert len(items) == config.size

    # Test multiple iterations yield same items
    assert items == list(dataset)


def test_course_schedule_answer():
    """Test the _can_finish method"""
    config = CourseScheduleConfig(seed=42)
    dataset = CourseScheduleDataset(config)

    prerequisites = [[0, 1]]
    assert dataset._can_finish(num_courses=2, prerequisites=prerequisites) == True

    # Direct cycle
    prerequisites = [[0, 1], [1, 0]]
    assert dataset._can_finish(num_courses=2, prerequisites=prerequisites) == False

    # Empty prerequisites
    prerequisites = []
    assert dataset._can_finish(num_courses=2, prerequisites=prerequisites) == True

    # Indirect cycle of length 3
    prerequisites = [[0, 1], [1, 2], [2, 0]]
    assert dataset._can_finish(num_courses=3, prerequisites=prerequisites) == False


def test_course_schedule_curriculum():
    curriculum = CourseScheduleCurriculum()

    base_value = {"size": 150, "seed": 1}

    base_cfg: CourseScheduleConfig = curriculum.generate_configuration(base_value)
    assert base_cfg.seed == 1
    assert base_cfg.size == 150
    assert base_cfg.min_num_courses == 5 and base_cfg.max_num_courses == 5
    assert base_cfg.min_num_prerequisites == 2 and base_cfg.max_num_prerequisites == 2
    assert base_cfg.min_cycle_length == 3 and base_cfg.max_cycle_length == 3

    # test incrementing attribute levels for num_terms & num_digits attributes
    curriculum.increment_attr_level("num_courses")
    curriculum.increment_attr_level("num_prerequisites")
    curriculum.increment_attr_level("cycle_length")
    increased_cfg = curriculum.generate_configuration(base_value)
    assert increased_cfg.min_num_courses == 5 and increased_cfg.max_num_courses == 50
    assert increased_cfg.min_num_prerequisites == 2 and increased_cfg.max_num_prerequisites == 4
    assert increased_cfg.min_cycle_length == 3 and increased_cfg.max_cycle_length == 5

    # test decrementing attribute level for num_digits again
    curriculum.decrement_attr_level("num_courses")
    partially_decreased_cfg = curriculum.generate_configuration(base_value)
    assert partially_decreased_cfg.min_num_courses == 5 and partially_decreased_cfg.max_num_courses == 5
    assert partially_decreased_cfg.min_num_prerequisites == 2 and partially_decreased_cfg.max_num_prerequisites == 4
    assert partially_decreased_cfg.min_cycle_length == 3 and partially_decreased_cfg.max_cycle_length == 5
