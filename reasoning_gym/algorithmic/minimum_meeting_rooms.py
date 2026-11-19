"""Minimum number of conference rooms required to schedule all meetings without overlap."""

from dataclasses import dataclass
from random import Random
from typing import Optional

from ..coaching import BaseCurriculum, RangeAttributeDefinition
from ..factory import ProceduralDataset, register_dataset

QUESTION_TEMPLATE = """Given a list of meeting time intervals [start, end], determine the minimum number of conference rooms required so that all meetings can take place without overlap.

If a meeting ends at time t, another meeting starting at time t can reuse the same room.

Now, find the minimum number of conference rooms required for the following {n} meetings:
{intervals}
"""

DATASET_NAME = "minimum_meeting_rooms"


@dataclass
class MinimumMeetingRoomsConfig:
    """Configuration for Minimum Meeting Rooms dataset generation."""

    min_n: int = 3  # Minimum number of meetings
    max_n: int = 15  # Maximum number of meetings
    min_start: int = 0  # Earliest possible start time
    max_end: int = 100  # Latest possible end time
    min_duration: int = 1  # Minimum meeting duration

    size: int = 500  # Virtual dataset size
    seed: Optional[int] = None

    def validate(self) -> None:
        """Validate configuration parameters."""
        assert self.min_n >= 1, "Minimum number of meetings must be at least 1"
        assert self.min_n <= self.max_n, "Minimum number of meetings must be <= maximum number of meetings"
        assert self.min_start >= 0, "Minimum start time must be non-negative"
        assert self.min_duration >= 1, "Minimum meeting duration must be at least 1"
        assert self.max_end >= self.min_start + self.min_duration, "max_end must be >= min_start + min_duration"


class MinimumMeetingRoomsDataset(ProceduralDataset):
    """Generates Minimum Meeting Rooms exercises with configurable difficulty."""

    def __init__(self, config: MinimumMeetingRoomsConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _get_answer(self, intervals: list[tuple[int, int]]) -> int:
        """Calculate the minimum number of conference rooms using a sweep-line algorithm."""
        if not intervals:
            return 0

        starts = sorted(start for start, _ in intervals)
        ends = sorted(end for _, end in intervals)

        rooms = 0
        max_rooms = 0
        end_ptr = 0

        for start in starts:
            while end_ptr < len(ends) and ends[end_ptr] <= start:
                rooms -= 1
                end_ptr += 1
            rooms += 1
            if rooms > max_rooms:
                max_rooms = rooms

        return max_rooms

    def __getitem__(self, idx: int) -> dict:
        """Generate a single Minimum Meeting Rooms question."""
        rng = Random(self.seed + idx)

        n = rng.randint(self.config.min_n, self.config.max_n)
        intervals: list[tuple[int, int]] = []
        for _ in range(n):
            start = rng.randint(self.config.min_start, self.config.max_end - self.config.min_duration)
            end = rng.randint(start + self.config.min_duration, self.config.max_end)
            intervals.append((start, end))

        answer = self._get_answer(intervals)
        formatted_intervals = ", ".join(f"[{s}, {e}]" for s, e in intervals)

        return {
            "question": QUESTION_TEMPLATE.format(n=n, intervals=formatted_intervals),
            "answer": str(answer),
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "intervals": intervals,
                "solution": answer,
                "n": n,
                "difficulty": {
                    "n": (self.config.min_n, self.config.max_n),
                    "max_end": self.config.max_end,
                },
            },
        }


class MinimumMeetingRoomsCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(MinimumMeetingRoomsCurriculum.__name__, MinimumMeetingRoomsConfig)

        self._define_attributes(
            RangeAttributeDefinition(
                name="n",
                levels=[3, 10, 25, 50],
                description="Number of meeting intervals",
                lower_field_name="min_n",
                upper_field_name="max_n",
                ensure_interval=True,
            ),
            RangeAttributeDefinition(
                name="max_end",
                levels=[20, 100, 500, 1000],
                description="Latest possible meeting end time",
                lower_field_name="max_end",
                upper_field_name="max_end",
            ),
        )


register_dataset(DATASET_NAME, MinimumMeetingRoomsDataset, MinimumMeetingRoomsConfig, MinimumMeetingRoomsCurriculum)
