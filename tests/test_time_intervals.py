from datetime import date, datetime

import pytest

from reasoning_gym.arithmetic import TimeIntervalsConfig, TimeIntervalsCurriculum, TimeIntervalsDataset


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


def test_oracle_answer():
    """Test that generated answer is marked correct"""
    config = TimeIntervalsConfig(seed=42, size=500)
    dataset = TimeIntervalsDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]

        metadata = item["metadata"]
        assert "start_time" in metadata
        assert "end_time" in metadata

        assert dataset.score_answer(item["answer"], item) == 1.0


def test_time_intervals_curriculum():
    curriculum = TimeIntervalsCurriculum()

    base_value = {"size": 150, "seed": 1}

    base_cfg: TimeIntervalsConfig = curriculum.generate_configuration(base_value)
    assert base_cfg.seed == 1
    assert base_cfg.size == 150
    assert base_cfg.max_time_difference_seconds == 60
    assert base_cfg.max_date_difference_days == 1

    # test incrementing attribute levels
    curriculum.increment_attr_level("max_time_difference_seconds")
    curriculum.increment_attr_level("max_date_difference_days")
    increased_cfg = curriculum.generate_configuration(base_value)
    assert increased_cfg.max_time_difference_seconds == 60 * 60
    assert increased_cfg.max_date_difference_days == 7

    # test decrementing attribute level
    curriculum.decrement_attr_level("max_time_difference_seconds")
    partially_decreased_cfg = curriculum.generate_configuration(base_value)
    assert partially_decreased_cfg.max_time_difference_seconds == 60
    assert partially_decreased_cfg.max_date_difference_days == 7
