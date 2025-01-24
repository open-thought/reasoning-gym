"""Least Common Multiple (LCM) task generator"""

from dataclasses import dataclass
from functools import reduce
from math import lcm
from random import Random
from typing import List, Optional, Tuple

from ..dataset import ProceduralDataset


@dataclass
class LCMConfig:
    """Configuration for LCM task generation"""

    min_numbers: int = 2  # Minimum numbers to find LCM of
    max_numbers: int = 2  # Maximum numbers to find LCM of
    min_value: int = 1  # Minimum value for each number
    max_value: int = 100  # Maximum value for each number (kept smaller than GCD default since LCM grows fast)
    seed: Optional[int] = None
    size: int = 500  # Virtual dataset size

    def validate(self):
        """Validate configuration parameters"""
        assert self.min_numbers >= 2, "min_numbers must be at least 2"
        assert self.max_numbers >= self.min_numbers, "max_numbers must be >= min_numbers"
        assert self.min_value >= 1, "min_value must be positive"
        assert self.max_value > self.min_value, "max_value must be > min_value"


class LCMDataset(ProceduralDataset):
    """Generates Least Common Multiple (LCM) tasks"""

    def __init__(self, config: LCMConfig):
        self.config = config
        self.config.validate()
        super().__init__(seed=config.seed, size=config.size)

    def _generate_numbers(self, rng: Random) -> Tuple[List[int], int]:
        """Generate a list of random positive integers and their LCM.
        Will try up to 3 times to find numbers with LCM < product."""

        def calculate_product(nums: List[int]) -> int:
            return reduce(lambda x, y: x * y, nums)

        for _ in range(3):  # Try up to 3 times to get LCM < product
            num_count = rng.randint(self.config.min_numbers, self.config.max_numbers)
            numbers = [rng.randint(self.config.min_value, self.config.max_value) for _ in range(num_count)]
            result = reduce(lcm, numbers)
            if result < calculate_product(numbers):
                return numbers, result

        # If we failed to find LCM < product after 3 tries, generate one final set
        num_count = rng.randint(self.config.min_numbers, self.config.max_numbers)
        numbers = [rng.randint(self.config.min_value, self.config.max_value) for _ in range(num_count)]
        result = reduce(lcm, numbers)
        return numbers, result

    def __getitem__(self, idx: int) -> dict:
        """Generate a single LCM task"""
        rng = Random(self.seed + idx)

        numbers, result = self._generate_numbers(rng)
        numbers_str = ", ".join(str(n) for n in numbers)

        return {
            "question": f"Find the Least Common Multiple (LCM) of these numbers: {numbers_str}",
            "answer": str(result),
            "metadata": {"numbers": numbers, "result": result},
        }


def lcm_dataset(
    min_numbers: int = 2,
    max_numbers: int = 2,
    min_value: int = 1,
    max_value: int = 100,
    seed: Optional[int] = None,
    size: int = 500,
) -> LCMDataset:
    """Create a LCMDataset with the given configuration."""
    config = LCMConfig(
        min_numbers=min_numbers,
        max_numbers=max_numbers,
        min_value=min_value,
        max_value=max_value,
        seed=seed,
        size=size,
    )
    return LCMDataset(config)
