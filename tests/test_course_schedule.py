"""Tests for Course Schedule puzzle generation"""

import pytest

from reasoning_gym.graphs.course_schedule import CourseScheduleConfig, CourseScheduleDataset


def test_course_schedule_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = CourseScheduleConfig(num_courses=-1)  # Negative not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = CourseScheduleConfig(num_courses=0)  # Zero not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = CourseScheduleConfig(max_num_prerequisites=-1)  # Negative not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = CourseScheduleConfig(max_num_prerequisites=0)  # Zero not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = CourseScheduleConfig(num_courses=3, max_num_prerequisites=5)  # max_num_prerequisites > num_courses
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
    config = CourseScheduleConfig(num_courses=15, size=10, seed=42)
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
        assert len(courses) == config.num_courses
        assert max(courses) == config.num_courses - 1
        assert len(prerequisites) <= config.max_num_prerequisites * config.num_courses
        assert all(len(prereq) == 2 for prereq in prerequisites)
        for course, prereq in prerequisites:
            assert course < config.num_courses
            assert prereq < config.num_courses
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
