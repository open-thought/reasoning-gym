"""Number sorting task generator"""

from dataclasses import dataclass
from random import Random
from typing import List, Optional, Tuple

from ..factory import ProceduralDataset, register_dataset


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

    def _format_number(self, num: float, decimals: int) -> str:
        """Format number with specified decimal places"""
        formatted = f"{num:.{decimals}f}"
        # Reparse to ensure exact decimal representation
        return f"{float(formatted):.{decimals}f}"

    def _generate_numbers(self, rng: Random) -> Tuple[List[float], List[str]]:
        """Generate list of numbers and their string representations"""
        count = rng.randint(self.config.min_numbers, self.config.max_numbers)
        decimals = rng.randint(self.config.min_decimals, self.config.max_decimals)

        numbers = []
        number_strs = []

        for _ in range(count):
            num = rng.uniform(self.config.min_value, self.config.max_value)
            num_str = self._format_number(num, decimals)
            # Reparse to ensure exact value
            num = float(num_str)
            numbers.append(num)
            number_strs.append(num_str)

        return numbers, number_strs

    def __getitem__(self, idx: int) -> dict:
        """Generate a single sorting task"""
        rng = Random(self.seed + idx)

        numbers, number_strs = self._generate_numbers(rng)

        # Generate both ascending and descending answers
        asc_numbers = sorted(numbers)
        desc_numbers = sorted(numbers, reverse=True)

        # Format answers as string lists
        decimals = len(number_strs[0].split(".")[-1]) if "." in number_strs[0] else 0
        asc_answer = [self._format_number(n, decimals) for n in asc_numbers]
        desc_answer = [self._format_number(n, decimals) for n in desc_numbers]

        # Randomly choose ascending or descending
        is_ascending = rng.choice([True, False])
        direction = "ascending" if is_ascending else "descending"
        answer = asc_answer if is_ascending else desc_answer

        return {
            "question": f"Sort these numbers in {direction} order: {', '.join(number_strs)}",
            "answer": str(answer),
            "metadata": {"original_numbers": number_strs, "direction": direction, "sorted_numbers": answer},
        }


register_dataset("number_sorting", NumberSortingDataset, NumberSortingConfig)
