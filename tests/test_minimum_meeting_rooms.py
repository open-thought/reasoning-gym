"""Tests for Minimum Meeting Rooms question generation."""

import pytest

import reasoning_gym
from reasoning_gym.algorithmic.minimum_meeting_rooms import (
    MinimumMeetingRoomsConfig,
    MinimumMeetingRoomsCurriculum,
    MinimumMeetingRoomsDataset,
)


def test_minimum_meeting_rooms_config_validation():
    """Test that invalid configs raise appropriate errors."""
    with pytest.raises(AssertionError):
        config = MinimumMeetingRoomsConfig(min_n=0)
        config.validate()

    with pytest.raises(AssertionError):
        config = MinimumMeetingRoomsConfig(min_n=-1)
        config.validate()

    with pytest.raises(AssertionError):
        config = MinimumMeetingRoomsConfig(min_n=10, max_n=5)
        config.validate()

    with pytest.raises(AssertionError):
        config = MinimumMeetingRoomsConfig(min_start=-1)
        config.validate()

    with pytest.raises(AssertionError):
        config = MinimumMeetingRoomsConfig(min_duration=0)
        config.validate()

    with pytest.raises(AssertionError):
        config = MinimumMeetingRoomsConfig(min_start=10, max_end=10, min_duration=1)
        config.validate()


def test_minimum_meeting_rooms_dataset_deterministic():
    """Test that dataset generates same items with same seed."""
    config = MinimumMeetingRoomsConfig(seed=42, size=10)
    dataset1 = MinimumMeetingRoomsDataset(config)
    dataset2 = MinimumMeetingRoomsDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_minimum_meeting_rooms_dataset_items():
    """Test basic properties of generated items and scoring."""
    dataset = reasoning_gym.create_dataset("minimum_meeting_rooms", max_n=10, size=10, seed=42)

    for i in range(len(dataset)):
        item = dataset[i]
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        assert "intervals" in item["metadata"]
        assert "solution" in item["metadata"]
        assert "n" in item["metadata"]

        intervals = item["metadata"]["intervals"]
        solution = item["metadata"]["solution"]
        assert len(intervals) == item["metadata"]["n"]
        assert 1 <= solution <= len(intervals)
        for start, end in intervals:
            assert start < end

        assert dataset.score_answer(item["answer"], item) == 1.0
        assert dataset.score_answer("0", item) == 0.0
        assert dataset.score_answer(None, item) == 0.0


def test_minimum_meeting_rooms_dataset_iteration():
    """Test that iteration respects dataset size."""
    config = MinimumMeetingRoomsConfig(size=5, seed=42)
    dataset = MinimumMeetingRoomsDataset(config)

    items = list(dataset)
    assert len(items) == config.size
    assert items == list(dataset)


def test_minimum_meeting_rooms_answer():
    """Verify the minimum meeting rooms calculation on known edge cases."""
    config = MinimumMeetingRoomsConfig(size=5, seed=42)
    dataset = MinimumMeetingRoomsDataset(config)

    # Empty
    assert dataset._get_answer([]) == 0

    # Back-to-back meetings reuse the same single room
    assert dataset._get_answer([(0, 5), (5, 10), (10, 15)]) == 1

    # All meetings overlap simultaneously
    assert dataset._get_answer([(1, 10), (2, 9), (3, 8)]) == 3

    # Classic partial overlap example
    assert dataset._get_answer([(0, 30), (5, 10), (15, 20)]) == 2


def test_minimum_meeting_rooms_curriculum():
    curriculum = MinimumMeetingRoomsCurriculum()

    base_value = {"size": 150, "seed": 1}

    base_cfg: MinimumMeetingRoomsConfig = curriculum.generate_configuration(base_value)
    assert base_cfg.seed == 1
    assert base_cfg.size == 150
    assert base_cfg.min_n == 3 and base_cfg.max_n == 10

    curriculum.increment_attr_level("n")
    increased_cfg = curriculum.generate_configuration(base_value)
    assert increased_cfg.min_n == 3 and increased_cfg.max_n == 25

    curriculum.decrement_attr_level("n")
    decreased_cfg = curriculum.generate_configuration(base_value)
    assert decreased_cfg.min_n == 3 and decreased_cfg.max_n == 10
