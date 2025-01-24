"""Number filtering task generator"""

from dataclasses import dataclass
from random import Random
from typing import List, Optional, Tuple

from ..factory import ProceduralDataset, register_dataset


@dataclass
class NumberFilteringConfig:
    """Configuration for number filtering task generation"""

    min_numbers: int = 3  # Minimum numbers in list
    max_numbers: int = 10  # Maximum numbers in list
    min_decimals: int = 0  # Minimum decimal places
    max_decimals: int = 4  # Maximum decimal places
    min_value: float = -100.0  # Minimum number value
    max_value: float = 100.0  # Maximum number value
    seed: Optional[int] = None
    size: int = 500  # Virtual dataset size

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.min_numbers > 0, "min_numbers must be positive"
        assert self.max_numbers >= self.min_numbers, "max_numbers must be >= min_numbers"
        assert self.min_decimals >= 0, "min_decimals must be non-negative"
        assert self.max_decimals >= self.min_decimals, "max_decimals must be >= min_decimals"
        assert self.max_value > self.min_value, "max_value must be > min_value"


class NumberFilteringDataset(ProceduralDataset):
    """Generates number filtering tasks"""

    def __init__(self, config: NumberFilteringConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _format_number(self, num: float, decimals: int) -> str:
        """Format a number with specified decimal places"""
        return f"{num:.{decimals}f}"

    def _generate_numbers(self, rng: Random) -> Tuple[List[float], List[str]]:
        """Generate list of numbers and their string representations"""
        count = rng.randint(self.config.min_numbers, self.config.max_numbers)
        numbers = []
        str_numbers = []

        for _ in range(count):
            num = rng.uniform(self.config.min_value, self.config.max_value)
            decimals = rng.randint(self.config.min_decimals, self.config.max_decimals)
            str_num = self._format_number(num, decimals)
            numbers.append(float(str_num))  # Convert back to simulate precision loss
            str_numbers.append(str_num)

        return numbers, str_numbers

    def __getitem__(self, idx: int) -> dict:
        """Generate a single number filtering task"""
        rng = Random(self.seed + idx)

        # Generate numbers and their string representations
        numbers, str_numbers = self._generate_numbers(rng)

        # Determine filter value between min and max of generated numbers
        min_val = min(numbers)
        max_val = max(numbers)
        filter_value = rng.uniform(min_val, max_val)
        decimals = rng.randint(self.config.min_decimals, self.config.max_decimals)
        filter_str = self._format_number(filter_value, decimals)
        filter_value = float(filter_str)  # Convert back to simulate precision loss

        # Randomly choose filter operation
        keep_larger = rng.choice([True, False])
        larger_smaller = "larger" if keep_larger else "smaller"
        keep_remove = "keep" if rng.choice([True, False]) else "remove"

        # Apply filter based on chosen operation
        if keep_remove == "keep":
            result = [n for n in numbers if (n > filter_value if keep_larger else n < filter_value)]
        else:  # remove
            result = [n for n in numbers if (n <= filter_value if keep_larger else n >= filter_value)]

        # Format results as strings with original precision
        result_strs = [str_numbers[numbers.index(n)] for n in result]

        return {
            "question": (
                f"{keep_remove.capitalize()} all numbers {larger_smaller} than {filter_str} "
                f"in this list: {str_numbers}"
            ),
            "answer": str(result_strs) if result_strs else "[]",
            "metadata": {
                "original_numbers": str_numbers,
                "filter_value": filter_str,
                "operation": f"{keep_remove}_{larger_smaller}",
                "result": result_strs,
            },
        }


register_dataset("number_filtering", NumberFilteringDataset, NumberFilteringConfig)
