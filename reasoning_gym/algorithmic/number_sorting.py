"""Number sorting task generator"""

import json
from dataclasses import dataclass
from random import Random
from typing import Any, Optional

import numpy as np

from ..coaching import BaseCurriculum, RangeAttributeDefinition, ScalarAttributeDefinition
from ..factory import ProceduralDataset, register_dataset

DATASET_NAME = "number_sorting"


@dataclass
class NumberSortingConfig:
    """Configuration for number sorting task generation"""

    min_numbers: int = 3  # Minimum numbers to sort
    max_numbers: int = 10  # Maximum numbers to sort
    min_decimals: int = 0  # Minimum decimal places
    max_decimals: int = 2  # Maximum decimal places
    min_value: float = -100.0  # Minimum value
    max_value: float = 100.0  # Maximum value
    seed: Optional[int] = None
    size: int = 500  # Virtual dataset size

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.min_numbers > 0, "min_numbers must be positive"
        assert self.min_numbers <= self.max_numbers, "max_numbers must be >= min_numbers"
        assert self.min_decimals >= 0, "min_decimals must be non-negative"
        assert self.min_decimals <= self.max_decimals, "max_decimals must be >= min_decimals"
        assert self.min_value < self.max_value, "max_value must be > min_value"


class NumberSortingDataset(ProceduralDataset):
    """Generates number sorting tasks"""

    def __init__(self, config: NumberSortingConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)
        self.added_instruction = """
Please follow the instruction below:
## 1. Let all your answers be a list of numbers. Instead of reporting your answer as -69, -13, 1, 7, 11, 43, 59, 61, use ['-69', '-13', '1', '7', '11', '43', '59', '61'] instead
## 2. Convert all numbers in the square brackets as strings. For example, ['-69', '-13', '1', '7', '11', '43', '59', '61']
"""

    def _generate_numbers(self, rng: Random, count: int) -> tuple[list[float], list[str]]:
        """Generate list of numbers and their string representations"""
        numbers = []
        number_strs = []

        for _ in range(count):
            num = rng.uniform(self.config.min_value, self.config.max_value)
            decimals = rng.randint(self.config.min_decimals, self.config.max_decimals)
            num = np.round(num, decimals)
            numbers.append(num)
            number_strs.append(str(num))

        return numbers, number_strs

    def __getitem__(self, idx: int) -> dict:
        """Generate a single sorting task"""
        rng = Random(self.seed + idx)

        count = rng.randint(self.config.min_numbers, self.config.max_numbers)
        numbers, number_strs = self._generate_numbers(rng, count)

        # Generate both ascending and descending answers
        asc_numbers = sorted(numbers)
        desc_numbers = sorted(numbers, reverse=True)

        # Format answers as string lists
        asc_answer = [str(n) for n in asc_numbers]
        desc_answer = [str(n) for n in desc_numbers]

        # Randomly choose ascending or descending
        is_ascending = rng.choice([True, False])
        direction = "ascending" if is_ascending else "descending"
        answer = asc_answer if is_ascending else desc_answer
        question = f"Sort these numbers in {direction} order: {', '.join(number_strs)}" + self.added_instruction

        return {
            "question": question,
            "answer": str(answer),
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "original_numbers": number_strs,
                "direction": direction,
                "sorted_numbers": answer,
                "numbers": count,
                "difficulty": {
                    "numbers": (self.config.min_numbers, self.config.max_numbers),
                    "decimals": (self.config.min_decimals, self.config.max_decimals),
                    "value": (self.config.min_value, self.config.max_value),
                },
            },
        }

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """Score the user's answer against the expected answer.

        Args:
            answer (Optional[str]): The user's answer string.
            entry (dict[str, Any]): The original dataset entry with the correct answer.

        Returns:
            float: 1.0 for a correct answer, 0.0 for incorrect.
        """
        if answer is None:
            return 0.0

        try:
            # Try to parse the user's answer as a JSON list first
            try:
                answer = answer.replace("'", '"')
                user_answer = json.loads(answer)
            except json.JSONDecodeError:
                return 0.0  # JSON parsing failed

            if not isinstance(user_answer, list):
                return 0.0

            # Get the expected answer
            try:
                expected_answer = json.loads(entry["answer"])
            except json.JSONDecodeError:
                # Fall back to eval if necessary
                expected_answer = eval(entry["answer"])

            # Check if the lists have the same length
            if len(user_answer) != len(expected_answer):
                return 0.0

            # Convert both answers to floats for comparison
            user_floats = [float(num) for num in user_answer]
            expected_floats = [float(num) for num in expected_answer]

            # First, verify the user's answer is properly sorted
            direction = entry["metadata"]["direction"]
            is_correctly_sorted = False

            if direction == "ascending":
                is_correctly_sorted = user_floats == sorted(user_floats)
            else:  # descending
                is_correctly_sorted = user_floats == sorted(user_floats, reverse=True)

            if not is_correctly_sorted:
                return 0.0

            # Check if the values are close enough (allowing for small rounding differences)
            tolerance = 1  # Increased tolerance to handle decimal differences
            for i in range(len(user_floats)):
                if abs(user_floats[i] - expected_floats[i]) > tolerance:
                    return 0.0

            return 1.0
        except Exception:
            # Any parsing error means the answer is incorrect
            return 0.0


class NumberSortingCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(NumberSortingCurriculum.__name__, NumberSortingConfig)

        # Define attributes
        self._define_attributes(
            RangeAttributeDefinition(
                name="numbers",
                levels=[10, 100, 500, 1000],
                description="How many numbers to sort",
                lower_field_name="min_numbers",
                upper_field_name="max_numbers",
                ensure_interval=True,
            ),
            RangeAttributeDefinition(
                name="decimals",
                levels=list(range(0, 8)),
                description="Number of decimal places",
                lower_field_name="min_decimals",
                upper_field_name="max_decimals",
                ensure_interval=True,
            ),
            ScalarAttributeDefinition(
                name="min_value",
                field_name="min_value",
                levels=[-100, -500, -1000, -10000],
                description="Minimum number value",
            ),
            ScalarAttributeDefinition(
                name="max_value",
                field_name="max_value",
                levels=[100, 500, 1000, 10000],
                description="Maximum number value",
            ),
        )


register_dataset(DATASET_NAME, NumberSortingDataset, NumberSortingConfig, NumberSortingCurriculum)
