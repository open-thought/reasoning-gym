import random
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import List, Optional

import pytz
from dateutil import parser

from ..factory import ProceduralDataset, register_dataset


@dataclass
class TimeIntervalsConfig:
    """Configuration for time interval calculation tasks"""

    min_time: time = time.min
    max_time: time = time.max
    max_time_difference_seconds: int = 24 * 60 * 60
    min_date: date = date(1900, 1, 1)
    max_date: date = date(3000, 1, 1)
    max_date_difference_days: int = 100
    task_types: List[str] = field(
        default_factory=lambda: ["time", "time_seconds", "time_ms", "date", "datetime", "datetime_tz"]
    )
    seed: Optional[int] = None
    size: int = 500

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.size > 0, "size must be positive"
        assert self.max_time_difference_seconds > 0, "max_time_difference_seconds must be positive"
        assert self.max_date_difference_days > 0, "max_date_difference_days must be positive"
        assert self.min_date < self.max_date, "min_date must be before max_date"


class TimeIntervalsDataset(ProceduralDataset):
    """Generates time interval calculation tasks with various formats and complexities"""

    TEMPLATES = [
        "What is the duration between {start} and {end}? Please answer in {format}.",
        "Calculate the time difference between {start} and {end}. Express the result in {format}.",
        "How much time elapsed from {start} to {end}? Give your answer in {format}.",
        "A meeting started at {start} and ended at {end}. How long was the meeting? Answer in {format}.",
        "A system operation started at {start} and completed at {end}. What was the operation duration? Answer in {format}.",
        "A database query started at {start} and ended at {end}. How long did the query take? Answer in {format}.",
        "A flight departed at {start} and arrived at {end}. How long was the flight? Answer in {format}.",
        "A video call started at {start} and ended at {end}. How long was the call? Answer in {format}.",
        "A system backup started at {start} and completed at {end}. What was the total backup duration? Answer in {format}.",
        "A conference call began at {start} and ended at {end}. How long was the conference? Answer in {format}.",
    ]

    TIME_FORMATS = [
        "%H:%M",
        "%H:%M:%S",
        "%H:%M:%S.%f",
    ]

    DATE_FORMATS = [
        "%Y-%m-%d",
        "%B %d, %Y",
        "%m/%d/%Y",
        "%A, %B %d, %Y",  # e.g. Monday, January 15, 2024
        "%a %b %d %Y",  # e.g. Mon Jan 15 2024
        "%d %B %Y",  # e.g. 15 January 2024
        "%Y-%m-%d (%A)",  # e.g. 2024-01-15 (Monday)
    ]

    DATETIME_FORMATS = [
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M %z",  # For UTC offset format
        "%Y-%m-%d %H:%M:%S %z",  # For UTC offset with seconds
        "%A, %B %d, %Y at %H:%M",  # e.g. Monday, January 15, 2024 at 14:30
        "%a %b %d %Y %H:%M:%S",  # e.g. Mon Jan 15 2024 14:30:45
        "%d %B %Y, %H:%M",  # e.g. 15 January 2024, 14:30
        "%d %B %Y, %H:%M %z",  # e.g. 15 January 2024, 14:30 +0000
        "%Y-%m-%d (%A) %H:%M:%S %z",  # e.g. 2024-01-15 (Monday) 14:30:45 +0000
    ]

    def __init__(self, config: TimeIntervalsConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def __getitem__(self, idx: int) -> dict:
        """Generate a single time interval calculation task"""
        item_rng = random.Random(self.seed + idx)

        # Randomly choose task type from config
        task_type = item_rng.choice(self.config.task_types)

        start_time, end_time, format_str, expected_format = self._generate_times(item_rng, task_type)

        template = item_rng.choice(self.TEMPLATES)
        question = template.format(start=start_time, end=end_time, format=expected_format)

        # Calculate the actual difference
        if isinstance(start_time, str):
            # Handle datetime strings with weekday names in parentheses
            start_time = start_time.split(" (")[0]  # Remove (Weekday) if present
            end_time = end_time.split(" (")[0]
            # Parse with UTC offset handling
            start_dt = parser.parse(start_time)
            end_dt = parser.parse(end_time)
        else:
            start_dt = start_time
            end_dt = end_time

        difference = end_dt - start_dt

        # Format the answer according to expected_format
        if expected_format == "HH:MM":
            total_seconds = difference.total_seconds()
            answer = f"{int(total_seconds // 3600):02d}:{int((total_seconds % 3600) // 60):02d}"
        elif expected_format == "HH:MM:SS":
            total_seconds = difference.total_seconds()
            answer = f"{int(total_seconds // 3600):02d}:{int((total_seconds % 3600) // 60):02d}:{int(total_seconds % 60):02d}"
        elif expected_format == "HH:MM:SS.mmm":
            total_seconds = difference.total_seconds()
            ms = int((total_seconds % 1) * 1000)
            answer = f"{int(total_seconds // 3600):02d}:{int((total_seconds % 3600) // 60):02d}:{int(total_seconds % 60):02d}.{ms:03d}"
        elif expected_format == "D days":
            answer = f"{difference.days} days"
        else:  # "D days, HH:MM" or "D days, HH:MM:SS"
            days = difference.days
            hours = difference.seconds // 3600
            minutes = (difference.seconds % 3600) // 60
            seconds = difference.seconds % 60
            if expected_format == "D days, HH:MM:SS":
                answer = f"{days} days, {hours:02d}:{minutes:02d}:{seconds:02d}"
            else:  # "D days, HH:MM"
                answer = f"{days} days, {hours:02d}:{minutes:02d}"

        return {
            "question": question,
            "answer": answer,
            "metadata": {
                "task_type": task_type,
                "start_time": start_dt,
                "end_time": end_dt,
                "format": format_str,
                "expected_format": expected_format,
            },
        }

    def _generate_times(self, rng: random.Random, task_type: str):
        """Generate start and end times based on task type"""
        if task_type.startswith("time"):
            if task_type == "time_ms":
                format_str = self.TIME_FORMATS[2]  # Get milliseconds format
                expected_format = "HH:MM:SS.mmm"
            else:
                format_str = next(f for f in self.TIME_FORMATS if f.count(":") == (2 if "seconds" in task_type else 1))
                expected_format = "HH:MM:SS" if "seconds" in task_type else "HH:MM"

            # Generate random start time
            start_hour = rng.randint(0, 23)
            start_minute = rng.randint(0, 59)
            start_second = rng.randint(0, 59)
            base = datetime.combine(date.today(), time(start_hour, start_minute, start_second))

            # Calculate seconds remaining until midnight
            seconds_until_midnight = ((24 - start_hour) * 3600) - (start_minute * 60) - start_second
            # Use the minimum of config max and seconds until midnight
            max_seconds = min(self.config.max_time_difference_seconds, seconds_until_midnight)
            diff_seconds = rng.randint(1, max_seconds) if max_seconds > 0 else 0

            if task_type == "time_ms":
                # Add microseconds for millisecond precision
                base = base.replace(microsecond=rng.randint(0, 999) * 1000)
                end_time = base + timedelta(seconds=diff_seconds, microseconds=rng.randint(0, 999) * 1000)
                # Format with exactly 3 decimal places for milliseconds
                start_time = base.strftime(format_str)[:-3]  # Remove extra microsecond digits
                end_time = end_time.strftime(format_str)[:-3]  # Remove extra microsecond digits
            else:
                start_time = base.strftime(format_str)
                end_time = (base + timedelta(seconds=diff_seconds)).strftime(format_str)

        elif task_type == "date":
            format_str = rng.choice(self.DATE_FORMATS)
            expected_format = "D days"  # Always return number of days for date tasks

            # Generate random start date within configured range, leaving room for end date
            max_date_difference_days = min(
                self.config.max_date_difference_days, (self.config.max_date - self.config.min_date).days
            )
            max_start_days = (self.config.max_date - self.config.min_date).days - max_date_difference_days
            start_days = rng.randint(0, max_start_days - 1)
            start_date = self.config.min_date + timedelta(days=start_days)

            # Ensure positive difference between dates
            diff_days = rng.randint(0, max_date_difference_days)
            end_date = start_date + timedelta(days=diff_days)

            start_time = start_date.strftime(format_str)
            end_time = end_date.strftime(format_str)

        else:  # datetime or datetime_tz
            format_str = rng.choice(self.DATETIME_FORMATS)
            # Choose between HH:MM and HH:MM:SS format for datetime answers
            expected_format = rng.choice(["D days, HH:MM", "D days, HH:MM:SS"])

            # Generate random start datetime
            days_range = (self.config.max_date - self.config.min_date).days
            start_days = rng.randint(0, days_range)
            start_hour = rng.randint(0, 23)
            start_minute = rng.randint(0, 59)
            start_second = rng.randint(0, 59)

            # Generate random time differences first
            diff_days = rng.randint(0, self.config.max_date_difference_days)
            diff_seconds = rng.randint(1, self.config.max_time_difference_seconds)

            if "%z" in format_str:
                # Use simpler timezone format with offset
                base = datetime.combine(
                    self.config.min_date + timedelta(days=start_days), time(start_hour, start_minute, start_second)
                )
                # Generate timezone offsets
                start_offset = rng.randint(-12, 12)
                end_offset = rng.randint(-12, 12)

                # Apply start timezone
                base = base.replace(tzinfo=pytz.FixedOffset(start_offset * 60))
                start_format = format_str.replace("%z", "%+05d" % (start_offset * 100))

                # Calculate end time and convert to end timezone
                end_dt = base + timedelta(days=diff_days, seconds=diff_seconds)
                end_dt = end_dt.replace(tzinfo=pytz.FixedOffset(end_offset * 60))
                end_format = format_str.replace("%z", "%+05d" % (end_offset * 100))

                # Format times with their respective timezone offsets
                start_time = base.strftime(start_format).rstrip()
                end_time = end_dt.strftime(end_format).rstrip()
            else:
                base = datetime.combine(
                    self.config.min_date + timedelta(days=start_days), time(start_hour, start_minute, start_second)
                )
                # For non-timezone aware times, both use same format
                start_time = base.strftime(format_str).rstrip()
                end_time = (base + timedelta(days=diff_days, seconds=diff_seconds)).strftime(format_str).rstrip()

        return start_time, end_time, format_str, expected_format

    def score_answer(self, answer: Optional[str], entry: dict) -> float:
        """Score an answer based on how close it is to the expected duration

        Returns a score between 0 and 1, with partial credit for answers that are
        close to correct in the appropriate units/format
        """
        if not answer:
            return 0.0

        expected = entry["answer"]
        task_type = entry["metadata"]["task_type"]

        try:
            if task_type == "date":
                # Parse "X days" format
                try:
                    actual = int(answer.strip().split()[0])  # Get number before "days"
                    expected = int(expected.strip().split()[0])
                    if actual == expected:
                        return 1.0
                    # Partial credit based on how close the day count is
                    max_diff = self.config.max_date_difference_days
                    diff = abs(actual - expected)
                    return max(0.0, 1.0 - (diff / max_diff))
                except (ValueError, IndexError):
                    return 0.0

            elif task_type.startswith("time"):
                # Parse times into total seconds for comparison
                def parse_time(t):
                    parts = t.strip().split(":")
                    seconds = int(parts[0]) * 3600 + int(parts[1]) * 60
                    if len(parts) > 2:
                        if "." in parts[2]:  # Has milliseconds
                            s, ms = parts[2].split(".")
                            seconds += int(s) + int(ms) / 1000
                        else:
                            seconds += int(parts[2])
                    return seconds

                actual_seconds = parse_time(answer)
                expected_seconds = parse_time(expected)

                if actual_seconds == expected_seconds:
                    return 1.0

                # Partial credit based on how close the times are
                max_diff = self.config.max_time_difference_seconds
                diff = abs(actual_seconds - expected_seconds)
                return max(0.0, 1.0 - (diff / max_diff))

            else:  # datetime or datetime_tz
                # Parse the complex format "X days, HH:MM" or "X days, HH:MM:SS"
                def parse_datetime(t):
                    days = int(t.split(" days,")[0])
                    time_part = t.split(",")[1].strip()
                    parts = time_part.split(":")
                    seconds = int(parts[0]) * 3600 + int(parts[1]) * 60
                    if len(parts) > 2:
                        seconds += int(parts[2])
                    return days * 86400 + seconds

                actual_seconds = parse_datetime(answer)
                expected_seconds = parse_datetime(expected)

                if actual_seconds == expected_seconds:
                    return 1.0

                # Partial credit based on total time difference
                max_diff = self.config.max_date_difference_days * 86400
                diff = abs(actual_seconds - expected_seconds)
                return max(0.0, 1.0 - (diff / max_diff))

        except (ValueError, IndexError):
            return 0.0  # Invalid format

        return 0.0


# Register the dataset
register_dataset("time_intervals", TimeIntervalsDataset, TimeIntervalsConfig)
