"""Merge all overlapping intervals.

A popular Leetcode problem:
https://leetcode.com/problems/merge-intervals/description/
"""

import ast
from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from ..coaching import BaseCurriculum, RangeAttributeDefinition, ScalarAttributeDefinition
from ..factory import ProceduralDataset, register_dataset

DATASET_NAME = "merge_intervals"

TOUCH_PROB = 0.2  # chance that a new interval starts exactly at an existing interval's end

QUESTION_TEMPLATE = """You are given a list of closed intervals, where each interval [start, end] contains every number from start to end (inclusive).

Merge all overlapping intervals and return the resulting list of non-overlapping intervals that covers exactly the same numbers as the input. Two intervals that touch at an endpoint (for example [1, 4] and [4, 7]) count as overlapping and must be merged.

Return the merged intervals sorted by their start value, as a list of [start, end] pairs, e.g. [[1, 6], [8, 10]]

Intervals: {intervals}
"""


@dataclass
class MergeIntervalsConfig:
    """Configuration for Merge Intervals dataset generation"""

    min_intervals: int = 2  # Minimum number of intervals in the input
    max_intervals: int = 10  # Maximum number of intervals in the input
    max_value: int = 50  # Interval endpoints are drawn from [0, max_value]
    max_interval_length: int = 10  # Maximum value of (end - start) for a single interval

    size: int = 500  # Virtual dataset size
    seed: Optional[int] = None

    def validate(self):
        """Validate configuration parameters"""
        assert self.size > 0, "size must be positive"
        assert 1 <= self.min_intervals <= self.max_intervals, "need 1 <= min_intervals <= max_intervals"
        assert self.max_interval_length >= 0, "max_interval_length must be non-negative"
        assert self.max_value >= self.max_interval_length, "max_value must be >= max_interval_length"


class MergeIntervalsDataset(ProceduralDataset):
    """Generates Merge Intervals exercises with configurable difficulty"""

    def __init__(self, config: MergeIntervalsConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _generate_intervals(self, rng: Random, n: int) -> list[list[int]]:
        """Generate n random intervals [start, end] with 0 <= start <= end <= max_value
        and end - start <= max_interval_length, in shuffled (unsorted) order.

        Most intervals are placed uniformly at random. With probability TOUCH_PROB an
        interval instead starts exactly where an earlier one ends, so the
        touching-endpoint case (which must merge) shows up regularly.
        """
        max_value = self.config.max_value
        max_len = self.config.max_interval_length

        intervals: list[list[int]] = []
        for _ in range(n):
            if intervals and rng.random() < TOUCH_PROB:
                start = rng.choice(intervals)[1]
                length = rng.randint(0, min(max_len, max_value - start))
            else:
                length = rng.randint(0, max_len)
                start = rng.randint(0, max_value - length)
            intervals.append([start, start + length])

        rng.shuffle(intervals)
        return intervals

    @staticmethod
    def _merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
        """Return the merged, sorted, non-overlapping intervals.

        Touching intervals ([1, 4] and [4, 7]) merge into [1, 7]. The input is not modified.
        """
        merged: list[list[int]] = []
        for start, end in sorted(intervals):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return merged

    @staticmethod
    def _parse_intervals(answer: str) -> Optional[list[list[int]]]:
        """Safely parse an answer like "[[1, 6], [8, 10]]"; return None if it is not a list of int pairs."""
        try:
            parsed = ast.literal_eval(answer.strip())
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            return None

        if not isinstance(parsed, (list, tuple)):
            return None
        result = []
        for pair in parsed:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                return None
            if not all(isinstance(x, int) and not isinstance(x, bool) for x in pair):
                return None
            result.append([pair[0], pair[1]])
        return result

    def __getitem__(self, idx: int) -> dict:
        """Generate a single Merge Intervals question"""
        rng = Random(self.seed + idx)

        n = rng.randint(self.config.min_intervals, self.config.max_intervals)
        intervals = self._generate_intervals(rng, n)
        merged = self._merge_intervals(intervals)

        return {
            "question": QUESTION_TEMPLATE.format(intervals=str(intervals)),
            "answer": str(merged),
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "intervals": intervals,
                "solution": merged,
                "num_intervals": n,
                "num_merged": len(merged),
                "difficulty": {
                    "num_intervals": (self.config.min_intervals, self.config.max_intervals),
                    "max_value": self.config.max_value,
                },
            },
        }

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """Score a model answer against the oracle solution.

        - 1.0: exactly the merged intervals, sorted by start (whitespace and tuple/list
          brackets are ignored).
        - 0.5: the right intervals but not sorted. The merge itself is correct, but the
          prompt explicitly asks for sorted output, so this gets partial credit only.
        - 0.0: anything else, including unparseable answers.
        """
        if not isinstance(answer, str) or not answer.strip():
            return 0.0

        parsed = self._parse_intervals(answer)
        if parsed is None:
            return 0.0

        solution = entry["metadata"]["solution"]
        if parsed == solution:
            return 1.0
        if sorted(parsed) == solution:
            return 0.5
        return 0.0


class MergeIntervalsCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(MergeIntervalsCurriculum.__name__, MergeIntervalsConfig)

        self._define_attributes(
            RangeAttributeDefinition(
                name="num_intervals",
                levels=[5, 10, 25, 50],
                description="Number of intervals in the input",
                lower_field_name="min_intervals",
                upper_field_name="max_intervals",
            ),
            ScalarAttributeDefinition(
                name="max_value",
                levels=[50, 100, 250, 500],
                description="Largest possible interval endpoint",
                field_name="max_value",
            ),
        )


register_dataset(DATASET_NAME, MergeIntervalsDataset, MergeIntervalsConfig, MergeIntervalsCurriculum)
