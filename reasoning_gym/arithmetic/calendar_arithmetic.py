import calendar
import math
import random
from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum, StrEnum, auto
from typing import Any, Optional

from ..factory import ProceduralDataset, register_dataset


class Weekday(Enum):
    MONDAY = auto()
    TUESDAY = auto()
    WEDNESDAY = auto()
    THURSDAY = auto()
    FRIDAY = auto()
    SATURDAY = auto()
    SUNDAY = auto()

    @classmethod
    def from_date(cls, d: date) -> "Weekday":
        return list(cls)[d.weekday()]

    @classmethod
    def random(cls, rng: random.Random) -> "Weekday":
        return list(cls)[rng.randint(0, 6)]

    @classmethod
    def __getitem__(cls, idx) -> "Weekday":
        return list(cls)[idx]

    @property
    def index(self) -> int:
        return self.value - 1

    def __str__(self) -> str:
        return self.name.capitalize()


class CalendarTask(StrEnum):
    WEEKDAY_OFFSET = "weekday_offset"
    WEEKDAY_OF_DATE = "weekday_of_date"
    WEEKDAY_OF_DATE_FROM_FIRST_DATE = "weekday_of_date_from_first_day"
    RECURRING_EVENT_CALCULATIONS = "recurring_event_day"
    COUNT_DAYS = "count_days"
    COUNT_BUSINESS_DAYS = "count_business_days"
    IS_LEAP_YEAR = "is_leap_year"


@dataclass
class CalendarArithmeticConfig:
    year: int = 2022
    tasks: Optional[list[str]] = None
    offset_upper_bound: int = 100
    leap_year_range: int = 200
    seed: Optional[int] = 42
    size: int = 500

    def __post_init__(self):
        if self.tasks is None:
            self.tasks = [task.value for task in CalendarTask]
        else:
            self.tasks = [task.lower() for task in self.tasks]
            valid_tasks = {task.value for task in CalendarTask}
            invalid_tasks = set(self.tasks) - valid_tasks
            if invalid_tasks:
                valid_task_list = ", ".join(sorted(valid_tasks))
                raise ValueError(
                    f"Invalid tasks: {', '.join(sorted(invalid_tasks))}. " f"Valid tasks are: {valid_task_list}"
                )

    def validate(self) -> None:
        """Validate the configuration parameters."""
        if not isinstance(self.year, int) or self.year <= 0:
            raise ValueError(f"year must be a positive integer, got {self.year}")

        if self.seed is not None and not isinstance(self.seed, int):
            raise ValueError(f"seed must be an integer or None, got {type(self.seed)}")

        if not isinstance(self.size, int) or self.size <= 0:
            raise ValueError(f"size must be a positive integer, got {self.size}")


class CalendarArithmeticDataset(ProceduralDataset):
    DAY_QUESTION_TEMPLATES = [
        "Answer with the weekday's name (e.g., Monday, Tuesday, etc.).",
        "Provide the full name of the weekday.",
        "State the weekday (Monday through Sunday).",
        "Give the weekday name in full.",
        "Reply with just the weekday name.",
        "Write out the full weekday name.",
        "Respond with the weekday (Monday-Sunday).",
        "Answer using the complete weekday name.",
        "Name the day of the week in full.",
    ]

    COUNT_QUESTION_TEMPLATES = [
        "Answer with a number.",
        "Provide the count as a number.",
        "Respond with just the number.",
        "Write the total number.",
        "Give the count numerically.",
        "State the amount as a number.",
        "Reply with the numerical value.",
        "Express your answer as a number.",
    ]

    def __init__(self, config: CalendarArithmeticConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

        self.task_handlers = {
            CalendarTask.WEEKDAY_OFFSET.value: self._weekday_offset,
            CalendarTask.WEEKDAY_OF_DATE.value: self._weekday_of_date,
            CalendarTask.WEEKDAY_OF_DATE_FROM_FIRST_DATE.value: self._weekday_of_date_from_first_day,
            CalendarTask.RECURRING_EVENT_CALCULATIONS.value: self._recurring_event_day,
            CalendarTask.COUNT_DAYS.value: self._count_days,
            CalendarTask.COUNT_BUSINESS_DAYS.value: self._count_business_days,
            CalendarTask.IS_LEAP_YEAR.value: self._is_leap_year,
        }

        self.tasks = [self.task_handlers[task] for task in self.config.tasks]

    def __getitem__(self, idx: int) -> dict:
        rng = random.Random(self.seed + idx)
        task = rng.choice(self.tasks)
        question, answer, metadata = task(rng)
        return {
            "question": question,
            "answer": str(answer),
            "metadata": metadata,
        }

    def _weekday_offset(self, rng: random.Random) -> tuple[str, str, dict]:
        """
        Task: Given a starting date and a day offset (which may be positive or negative),
        ask what day of the week it will be.
        Examples:
         - "If today is Wednesday, March 13, 2024, what day of the week will it be in 10 days? Answer with the weekday's name."
         - "If today is Wednesday, March 13, 2024, what day of the week was it 10 days ago? Answer with the weekday's name."
        """
        year = self.config.year
        start_date = self._random_date_for_year(rng, year)
        offset = rng.randint(1, self.config.offset_upper_bound)
        sign = rng.choice([-1, 1])
        offset_days = sign * offset
        target_date = start_date + timedelta(days=offset_days)
        target_weekday = target_date.strftime("%A")

        date_str = f"{start_date.strftime('%A')}, {start_date.strftime('%B')} {start_date.day}, {start_date.year}"
        if offset_days >= 0:
            templates = [
                f"If today is {date_str}, what day of the week will it be in {offset_days} days? ",
                f"Starting from {date_str}, which weekday falls after a {offset_days}-day jump? ",
                f"Count forward {offset_days} days from {date_str} - what's the weekday? ",
            ]
        else:
            templates = [
                f"If today is {date_str}, what day of the week was it {abs(offset_days)} days ago? ",
                f"Starting from {date_str}, which weekday was it {abs(offset_days)} days before? ",
                f"Count backward {abs(offset_days)} days from {date_str} - what's the weekday? ",
            ]

        question = rng.choice(templates) + rng.choice(self.DAY_QUESTION_TEMPLATES)
        metadata = {
            "task": CalendarTask.WEEKDAY_OFFSET.value,
            "start_date": start_date.isoformat(),
            "offset_days": offset_days,
            "target_date": target_date.isoformat(),
        }
        return question, target_weekday, metadata

    def _weekday_of_date(self, rng: random.Random) -> tuple[str, str, dict]:
        """
        task: Ask what day of the week a given date was.
        example:
          "What day of the week was January 15, 2024?
           Answer with the weekday's name."
        """
        year = self.config.year
        target_date = self._random_date_for_year(rng, year)
        answer_weekday = target_date.strftime("%A")
        templates = [
            f"What day of the week was {target_date.strftime('%B')} {target_date.day}, {year}?",
            f"On which weekday did {target_date.strftime('%B')} {target_date.day}, {year} fall?",
            f"Name the day of the week for {target_date.strftime('%m/%d/%Y')}.",
        ]

        question = f"{rng.choice(templates)} {rng.choice(self.DAY_QUESTION_TEMPLATES)}"
        metadata = {
            "task": CalendarTask.WEEKDAY_OF_DATE.value,
            "target_date": target_date.isoformat(),
        }
        return question, answer_weekday, metadata

    def _weekday_of_date_from_first_day(self, rng: random.Random) -> tuple[str, str, dict]:
        """
        task: Given an hypothetical weekday for January 1, ask what weekday a later date in the year falls on.
        example:
         "If the first day of the year was a Monday, what day of the week will December 31 be?
          Answer with the weekday's name."
        """
        year = self.config.year
        first_day = Weekday.random(rng)
        first_day_index = first_day.index
        # Ensure target date is not January 1.
        year_start = date(year, 1, 1)
        year_end = date(year, 12, 31)
        max_delta = timedelta(days=self.config.offset_upper_bound)
        max_date = min(year_start + max_delta, year_end)
        while True:
            target_date = self._random_date_between(rng, year_start, max_date)
            if target_date != date(year, 1, 1):
                break
        delta_days = (target_date - date(year, 1, 1)).days
        answer_index = (first_day_index + delta_days) % 7
        answer_weekday = Weekday(answer_index + 1)

        templates = [
            f"If the first day of the year was a {first_day}, what day of the week will "
            f"{target_date.strftime('%B')} {target_date.day} be? ",
            f"Given that January 1 fell on a {first_day}, which weekday occurs on "
            f"{target_date.strftime('%B')} {target_date.day}? ",
            f"In a year where {first_day} is January 1st, name the weekday of "
            f"{target_date.strftime('%B')} {target_date.day}. ",
        ]

        question = rng.choice(templates) + rng.choice(self.DAY_QUESTION_TEMPLATES)
        metadata = {
            "task": CalendarTask.WEEKDAY_OF_DATE_FROM_FIRST_DATE.value,
            "year": year,
            "first_day": str(first_day),
            "target_date": target_date.isoformat(),
            "delta_days": delta_days,
        }
        return question, answer_weekday, metadata

    def _recurring_event_day(self, rng: random.Random) -> tuple[str, str, dict]:
        """
        task: For a recurring event defined by an ordinal weekday pattern in a month,
        ask on which day of the month the event occurs.
        example:
         "If a meeting is scheduled on the second Tuesday of May 2024, on which day does it fall?
          Answer with a number."
        """
        year = self.config.year
        month = rng.randint(1, 12)
        ordinals = ["first", "second", "third", "fourth", "last"]
        ordinal = rng.choice(ordinals)
        weekday = Weekday.random(rng)
        month_name = calendar.month_name[month]
        _, last_day = calendar.monthrange(year, month)

        if ordinal != "last":
            ordinal_number = {"first": 1, "second": 2, "third": 3, "fourth": 4}[ordinal]
            count = 0
            event_day = None
            for day in range(1, last_day + 1):
                d = date(year, month, day)
                if d.strftime("%A") == str(weekday):
                    count += 1
                    if count == ordinal_number:
                        event_day = day
                        break
            if event_day is None:
                # This should rarely happen but in some months the ordinal may not exist.
                event_day = -1
        else:
            event_day = None
            for day in range(last_day, 0, -1):
                d = date(year, month, day)
                if d.strftime("%A") == str(weekday):
                    event_day = day
                    break
            if event_day is None:
                event_day = -1

        templates = [
            f"If a meeting is scheduled on the {ordinal} {weekday} of {month_name} {year}, on which day of the month does it occur? ",
            f"In {month_name} {year}, if an event recurs on the {ordinal} {weekday}, what is the date (day of the month) of the event? ",
            f"Determine the day of the month for the {ordinal} {weekday} in {month_name} {year}. ",
        ]
        question = (
            rng.choice(templates)
            + rng.choice(self.COUNT_QUESTION_TEMPLATES)
            + " Answer with -1 if the ordinal does not exist in the month."
        )
        metadata = {
            "task": CalendarTask.RECURRING_EVENT_CALCULATIONS.value,
            "year": year,
            "month": month,
            "ordinal": ordinal,
            "weekday": str(weekday),
        }
        return question, str(event_day), metadata

    def _count_days(self, rng: random.Random) -> tuple[str, str, dict]:
        """
        task: Ask how many times a given weekday occurs in a specified range.
        example:
           "How many days are there between March 1, 2024 and March 15, 2024?
           Answer with a number."
        """
        year = self.config.year
        year_start = date(year, 1, 1)
        year_end = date(year, 12, 31)
        start_date = self._random_date_between(rng, year_start, year_end)
        max_delta = timedelta(days=self.config.offset_upper_bound)
        end_date = self._random_date_between(rng, start_date, min(year_end, start_date + max_delta))
        weekday = Weekday.random(rng)

        def count_weekday_between(d1: date, d2: date, weekday: str) -> int:
            days = (d2 - d1).days + 1
            return sum(1 for i in range(days) if (d1 + timedelta(days=i)).strftime("%A") == weekday)

        count = count_weekday_between(start_date, end_date, str(weekday))

        templates = [
            f"How many {weekday}s are there from {start_date.strftime('%A, %B')} {start_date.day}, {year} to "
            f"{end_date.strftime('%A, %B')} {end_date.day}, {year} (inclusive of both dates)? ",
            f"Count the occurrences of {weekday} from {start_date.strftime('%A, %B')} {start_date.day} "
            f"to {end_date.strftime('%A, %B')} {end_date.day}, {year} (including both start and end dates). ",
            f"Between {start_date.strftime('%A, %B')} {start_date.day}, {year} and "
            f"{end_date.strftime('%A, %B')} {end_date.day}, {year} "
            f"(counting both dates), how many times does {weekday} occur? ",
        ]

        question = rng.choice(templates) + rng.choice(self.COUNT_QUESTION_TEMPLATES)
        metadata = {
            "task": CalendarTask.COUNT_DAYS.value,
            "year": year,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        }
        return question, str(count), metadata

    def _count_business_days(self, rng: random.Random) -> tuple[str, str, dict]:
        """
        task: Count the number of business days (Monday-Friday) between two dates.
        example:
          "How many business days (Monday-Friday) are there between March 1, 2024 and March 15, 2024?
           Answer with a number."
        """
        year = self.config.year
        year_start = date(year, 1, 1)
        year_end = date(year, 12, 31)
        start_date = self._random_date_between(rng, year_start, year_end)
        max_delta = timedelta(days=self.config.offset_upper_bound)
        end_date = self._random_date_between(rng, start_date, start_date + max_delta)

        count = 0

        def business_days_between(d1: date, d2: date) -> int:
            days = (d2 - d1).days + 1
            weeks, remainder = divmod(days, 7)
            count = weeks * 5
            start_weekday = d1.weekday()
            for i in range(remainder):
                if (start_weekday + i) % 7 < 5:
                    count += 1
            return count

        count = business_days_between(start_date, end_date)

        templates = [
            f"How many business days (Monday-Friday) are there from "
            f"{start_date.strftime('%A, %B')} {start_date.day}, {year} to "
            f"{end_date.strftime('%A, %B')} {end_date.day}, {year} "
            f"(inclusive of both dates)? ",
            f"Count the weekdays (excluding weekends) from "
            f"{start_date.strftime('%A, %B')} {start_date.day} to "
            f"{end_date.strftime('%A, %B')} {end_date.day}, {year} "
            f"(including both start and end dates). ",
            f"Between {start_date.strftime('%A, %B')} {start_date.day}, {year} and "
            f"{end_date.strftime('%A, %B')} {end_date.day}, {year} "
            f"(counting both dates), what's the total count of business days "
            f"(Monday through Friday)? ",
        ]

        question = rng.choice(templates) + rng.choice(self.COUNT_QUESTION_TEMPLATES)
        metadata = {
            "task": CalendarTask.COUNT_BUSINESS_DAYS.value,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        }
        return question, str(count), metadata

    def _is_leap_year(self, rng: random.Random) -> tuple[str, str, dict]:
        """
        task: Given a year, determine whether it is a leap year.
        example:
         "Is 2024 a leap year? Answer with Yes or No."
        """
        semirange = self.config.leap_year_range // 2
        year = rng.randint(self.config.year - semirange, self.config.year + semirange)
        is_leap = calendar.isleap(year)
        answer = "Yes" if is_leap else "No"
        templates = [
            f"Determine if the year {year} is a leap year. ",
            f"Is {year} a leap year? ",
            f"Tell me whether {year} is a leap year. ",
        ]
        question = rng.choice(templates) + "Answer with Yes or No."
        metadata = {
            "task": CalendarTask.IS_LEAP_YEAR.value,
            "year": year,
            "is_leap": is_leap,
        }
        return question, answer, metadata

    def _random_date_for_year(self, rng: random.Random, year: int) -> date:
        """Return a random date within the given year."""
        month = rng.randint(1, 12)
        _, last_day = calendar.monthrange(year, month)
        day = rng.randint(1, last_day)
        return date(year, month, day)

    def _random_date_between(self, rng: random.Random, start_date: date, end_date: date) -> date:
        """
        Return a random date between start_date and end_date (inclusive).
        Assumes start_date <= end_date.
        """
        if start_date > end_date:
            raise ValueError("start_date must be <= end_date")
        delta = (end_date - start_date).days
        random_days = rng.randint(0, delta)
        return start_date + timedelta(days=random_days)

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        # we suppose the answer is the last occurence of the expected answer type
        if answer is None:
            return 0.0

        oracle_answer = entry["answer"]
        task = entry["metadata"]["task"]

        if task in {
            CalendarTask.WEEKDAY_OFFSET.value,
            CalendarTask.WEEKDAY_OF_DATE_FROM_FIRST_DATE.value,
            CalendarTask.WEEKDAY_OF_DATE.value,
        }:
            if not answer:
                return 0.0

            answer = answer.strip()
            oracle_answer = oracle_answer
            weekdays = {d.name.title() for d in Weekday}

            if answer == oracle_answer:
                return 1.0

            if answer in weekdays:
                return 0.1

            if answer.title() in weekdays:
                return 0.05

            if answer.title() not in weekdays:
                return 0.0

            return 0.0

        # denser reward for numerical tasks
        elif task in {
            CalendarTask.COUNT_BUSINESS_DAYS.value,
            CalendarTask.COUNT_DAYS.value,
            CalendarTask.RECURRING_EVENT_CALCULATIONS.value,
        }:
            try:
                ans_num = int(answer.strip())
                oracle_num = int(oracle_answer.strip())

                if oracle_num == 0:
                    return 1.0 if ans_num == 0 else 0.0

                relative_error = abs(ans_num - oracle_num) / oracle_num
                return max(0.0, math.exp(-5 * relative_error))

            except (ValueError, AttributeError):
                return 0.0

        elif task == CalendarTask.IS_LEAP_YEAR.value:
            if answer.strip().lower() == oracle_answer.lower():
                return 1.0
            return 0.0

        return 0.0


register_dataset("calendar_arithmetic", CalendarArithmeticDataset, CalendarArithmeticConfig)
