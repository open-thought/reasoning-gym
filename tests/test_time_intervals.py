from datetime import date, datetime

import pytest

from reasoning_gym.arithmetic import TimeIntervalsConfig, TimeIntervalsDataset


def test_time_intervals_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = TimeIntervalsConfig(size=0)
        config.validate()

    with pytest.raises(AssertionError):
        config = TimeIntervalsConfig(max_time_difference_seconds=0)
        config.validate()

    with pytest.raises(AssertionError):
        config = TimeIntervalsConfig(max_date_difference_days=0)
        config.validate()

    with pytest.raises(AssertionError):
        config = TimeIntervalsConfig(min_date=date(2024, 1, 1), max_date=date(2023, 1, 1))
        config.validate()


def test_time_intervals_deterministic():
    """Test that dataset generates same items with same seed"""
    config = TimeIntervalsConfig(seed=42, size=10)
    dataset1 = TimeIntervalsDataset(config)
    dataset2 = TimeIntervalsDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_time_intervals_items():
    """Test basic properties of generated items"""
    config = TimeIntervalsConfig(
        size=100,
        seed=42,
        max_time_difference_seconds=3600,  # 1 hour max
        max_date_difference_days=10,
    )
    dataset = TimeIntervalsDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item
        assert "task_type" in item["metadata"]
        assert "start_time" in item["metadata"]
        assert "end_time" in item["metadata"]


def test_time_intervals_scoring():
    """Test the answer scoring functionality"""
    config = TimeIntervalsConfig(seed=42)
    dataset = TimeIntervalsDataset(config)

    # Generate a sample item
    item = dataset[0]

    # Test exact match
    assert dataset.score_answer(item["answer"], item) == 1.0

    # Test empty/None answers
    assert dataset.score_answer(None, item) == 0.0
    assert dataset.score_answer("", item) == 0.0

    # Test invalid format
    assert dataset.score_answer("invalid", item) == 0.0

    # Test close but not exact answers
    task_type = item["metadata"]["task_type"]
    if task_type == "date":
        expected = int(item["answer"])
        # Test answer off by 1 day
        score = dataset.score_answer(str(expected + 1), item)
        assert 0 < score < 1
    elif task_type.startswith("time"):
        # Test answer off by a few minutes
        if ":" in item["answer"]:
            parts = item["answer"].split(":")
            hours = int(parts[0])
            minutes = (int(parts[1]) + 5) % 60  # Add 5 minutes
            modified = f"{hours:02d}:{minutes:02d}"
            if len(parts) > 2:
                modified += ":" + parts[2]
            score = dataset.score_answer(modified, item)
            assert 0 < score < 1


def test_time_format_patterns():
    """Test that generated times match expected formats"""
    config = TimeIntervalsConfig(seed=42, size=500)
    dataset = TimeIntervalsDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]

        start_dt = item["metadata"]["start_time"]
        end_dt = item["metadata"]["end_time"]

        # Verify both are datetime objects
        assert isinstance(start_dt, datetime)
        assert isinstance(end_dt, datetime)

        # Verify end is after start
        assert end_dt >= start_dt, item["question"]
        assert dataset.score_answer(item["answer"], item) == 1.0
