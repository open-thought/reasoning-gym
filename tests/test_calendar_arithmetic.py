import calendar
import math
from datetime import date

import pytest

from reasoning_gym.arithmetic import CalendarArithmeticConfig, CalendarArithmeticDataset

WEEKDAYS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

WEEKDAY_TASKS = {
    "weekday_offset",
    "weekday_of_date_from_first_day",
    "weekday_of_date",
}
NUMERIC_TASKS = {
    "count_days",
    "count_business_days",
}
DAY_TASKS = {"recurring_event_day"}
BOOLEAN_TASKS = {"is_leap_year"}
CALENDAR_TASKS = WEEKDAY_TASKS | NUMERIC_TASKS | DAY_TASKS | BOOLEAN_TASKS


def test_calendar_config_validation():
    """Test that invalid CalendarArithmeticConfig parameters raise appropriate errors."""
    with pytest.raises(ValueError):
        config = CalendarArithmeticConfig(year=0)
        config.validate()

    with pytest.raises(ValueError):
        config = CalendarArithmeticConfig(size=0)
        config.validate()

    with pytest.raises(ValueError):
        config = CalendarArithmeticConfig(seed="not_an_int")
        config.validate()

    with pytest.raises(ValueError):
        config = CalendarArithmeticConfig(tasks=["invalid_task"])


def test_calendar_deterministic():
    """Test that a dataset with a fixed seed produces the same items."""
    config = CalendarArithmeticConfig(year=2024, seed=42, size=10)
    ds1 = CalendarArithmeticDataset(config)
    ds2 = CalendarArithmeticDataset(config)

    for i in range(len(ds1)):
        assert ds1[i] == ds2[i]


def test_calendar_item_structure():
    """Test that dataset items have the correct structure and fields."""
    config = CalendarArithmeticConfig(year=2024, seed=42, size=50)
    dataset = CalendarArithmeticDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        assert isinstance(item, dict)
        assert all(key in item for key in ["question", "answer", "metadata"])

        assert isinstance(item["question"], str) and len(item["question"]) > 0
        assert isinstance(item["answer"], str) and len(item["answer"]) > 0
        assert "task" in item["metadata"]
        assert item["metadata"]["task"] in CALENDAR_TASKS


def test_calendar_answer_format():
    """Test that answers have the correct format based on task type."""
    config = CalendarArithmeticConfig(year=2024, seed=42, size=100)
    dataset = CalendarArithmeticDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        task = item["metadata"]["task"]
        answer = item["answer"]

        if task in WEEKDAY_TASKS:
            assert answer in WEEKDAYS

        elif task in NUMERIC_TASKS:
            try:
                num = int(answer)
                assert num >= 0, f"task {task} produced a negative count: {num}"
            except ValueError:
                pytest.fail(f"task {task} produced a non-integer answer: {answer}")

        elif task in BOOLEAN_TASKS:
            assert answer in ["Yes", "No"]

        elif task in DAY_TASKS:
            try:
                num = int(answer)
                year = item["metadata"]["year"]
                month = item["metadata"]["month"]
                _, last_day = calendar.monthrange(year, month)
                assert 1 <= num <= last_day
            except ValueError:
                pytest.fail(f"task {task} produced a day outside expected range (1-{last_day}): {answer}")


def test_scoring_function():
    """Test scoring function for different answer types."""
    config = CalendarArithmeticConfig(year=2024, seed=42, size=1)
    dataset = CalendarArithmeticDataset(config)

    weekday_item = {"answer": "Monday", "metadata": {"task": "weekday_offset"}}

    assert dataset.score_answer("Monday", weekday_item) == 1.0
    assert dataset.score_answer("Tuesday", weekday_item) == 0.1
    assert dataset.score_answer("It is Monday", weekday_item) == 0.0
    assert dataset.score_answer("no weekday here", weekday_item) == 0.0
    assert dataset.score_answer(None, weekday_item) == 0.0

    numeric_item = {"answer": "10", "metadata": {"task": "count_business_days"}}
    assert dataset.score_answer("10", numeric_item) == 1.0
    assert dataset.score_answer("15", numeric_item) == pytest.approx(math.exp(-5 * 0.5))
    assert dataset.score_answer("no number", numeric_item) == 0.0
    assert dataset.score_answer(None, numeric_item) == 0.0

    boolean_item = {"answer": "Yes", "metadata": {"task": "is_leap_year"}}
    assert dataset.score_answer("Yes", boolean_item) == 1.0
    assert dataset.score_answer("yes", boolean_item) == 1.0
    assert dataset.score_answer("nyes", boolean_item) == 0.0
    assert dataset.score_answer(None, boolean_item) == 0.0


def test_calendar_date_consistency():
    """Test that dates in metadata are consistent with config year."""
    config = CalendarArithmeticConfig(year=2024, seed=42, size=50)
    dataset = CalendarArithmeticDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        task = item["metadata"]["task"]

        if task == "weekday_offset":
            start_date = date.fromisoformat(item["metadata"]["start_date"])
            assert start_date.year == config.year

        elif task in {"weekday_of_date_from_first_day", "weekday_of_date"}:
            target_date = date.fromisoformat(item["metadata"]["target_date"])
            assert target_date.year == config.year

        elif task in {"count_business_days", "count_days"}:
            start_date = date.fromisoformat(item["metadata"]["start_date"])
            end_date = date.fromisoformat(item["metadata"]["end_date"])
            assert start_date.year == config.year
            assert end_date.year == config.year

        elif task == "recurring_event_day":
            meta_year = item["metadata"]["year"]
            month = item["metadata"]["month"]
            answer = int(item["answer"])
            assert meta_year == config.year
            assert 1 <= month <= 12
            if answer != -1:
                _, last_day = calendar.monthrange(meta_year, month)
                assert 1 <= answer <= last_day

        elif task == "is_leap_year":
            year = item["metadata"]["year"]
            assert config.year - 200 <= year <= config.year + 200
            is_leap_metadata = item["metadata"]["is_leap"]
            computed_is_leap = calendar.isleap(year)
            assert is_leap_metadata == computed_is_leap


def test_calendar_iteration():
    """Test that dataset iteration works correctly and is deterministic."""
    config = CalendarArithmeticConfig(year=2024, seed=42, size=5)
    dataset = CalendarArithmeticDataset(config)

    items = [item for item in dataset]
    assert len(items) == config.size

    first_iter = list(dataset)
    second_iter = list(dataset)
    assert first_iter == second_iter


def test_task_case_sensitivity():
    """Test that task names are case-insensitive."""
    tasks = ["WEEKDAY_OFFSET", "Count_Business_Days"]
    config = CalendarArithmeticConfig(tasks=tasks, size=10)
    dataset = CalendarArithmeticDataset(config)

    for item in dataset:
        assert item["metadata"]["task"] in [t.lower() for t in tasks]
