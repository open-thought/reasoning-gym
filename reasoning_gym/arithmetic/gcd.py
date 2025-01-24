"""Greatest Common Divisor (GCD) task generator"""

from dataclasses import dataclass
from functools import reduce
from math import gcd
from random import Random
from typing import List, Optional, Tuple

from ..factory import ProceduralDataset, register_dataset


@dataclass
class GCDConfig:
    """Configuration for GCD task generation"""

    min_numbers: int = 2  # Minimum numbers to find GCD of
    max_numbers: int = 2  # Maximum numbers to find GCD of
    min_value: int = 1  # Minimum value for each number
    max_value: int = 1000  # Maximum value for each number
    seed: Optional[int] = None
    size: int = 500  # Virtual dataset size

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.min_numbers >= 2, "min_numbers must be at least 2"
        assert self.max_numbers >= self.min_numbers, "max_numbers must be >= min_numbers"
        assert self.min_value >= 1, "min_value must be positive"
        assert self.max_value > self.min_value, "max_value must be > min_value"


class GCDDataset(ProceduralDataset):
    """Generates Greatest Common Divisor (GCD) tasks"""

    def __init__(self, config: GCDConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _generate_numbers(self, rng: Random) -> Tuple[List[int], int]:
        """Generate a list of random positive integers and their GCD.
        Will try up to 3 times to find numbers with GCD > 1."""

        # Try up to 3 times to get GCD > 1
        for _ in range(3):
            num_count = rng.randint(self.config.min_numbers, self.config.max_numbers)
            numbers = [rng.randint(self.config.min_value, self.config.max_value) for _ in range(num_count)]
            result = reduce(gcd, numbers)
            if result > 1:
                break

        # Return the last generated numbers, whether they met the criteria or not
        return numbers, result

    def __getitem__(self, idx: int) -> dict:
        """Generate a single GCD task"""
        rng = Random(self.seed + idx)

        numbers, result = self._generate_numbers(rng)
        numbers_str = ", ".join(str(n) for n in numbers)

        return {
            "question": f"Find the Greatest Common Divisor (GCD) of these numbers: {numbers_str}",
            "answer": str(result),
            "metadata": {"numbers": numbers, "result": result},
        }


register_dataset("gcd", GCDDataset, GCDConfig)
