"""Least Common Multiple (LCM) task generator"""
from dataclasses import dataclass
from random import Random
from typing import List, Optional, Tuple
from math import lcm
from functools import reduce


@dataclass
class LCMConfig:
    """Configuration for LCM task generation"""
    min_numbers: int = 2       # Minimum numbers to find LCM of
    max_numbers: int = 2       # Maximum numbers to find LCM of
    min_value: int = 1        # Minimum value for each number
    max_value: int = 100      # Maximum value for each number (kept smaller than GCD default since LCM grows fast)
    seed: Optional[int] = None
    size: int = 500          # Virtual dataset size

    def validate(self):
        """Validate configuration parameters"""
        assert self.min_numbers >= 2, "min_numbers must be at least 2"
        assert self.max_numbers >= self.min_numbers, "max_numbers must be >= min_numbers"
        assert self.min_value >= 1, "min_value must be positive"
        assert self.max_value > self.min_value, "max_value must be > min_value"


class LCMDataset:
    """Generates Least Common Multiple (LCM) tasks"""

    def __init__(self, config: LCMConfig):
        self.config = config
        self.config.validate()
        self.seed = config.seed if config.seed is not None else Random().randint(0, 2**32)

    def __len__(self) -> int:
        return self.config.size

    def __iter__(self):
        self._current_idx = 0
        return self

    def __next__(self):
        if self._current_idx >= self.config.size:
            raise StopIteration
        item = self[self._current_idx]
        self._current_idx += 1
        return item

    def _generate_numbers(self, rng: Random) -> List[int]:
        """Generate a list of random positive integers"""
        num_count = rng.randint(self.config.min_numbers, self.config.max_numbers)
        return [rng.randint(self.config.min_value, self.config.max_value) 
                for _ in range(num_count)]

    def _calculate_lcm(self, numbers: List[int]) -> int:
        """Calculate the LCM of a list of numbers"""
        return reduce(lcm, numbers)

    def __getitem__(self, idx: int) -> dict:
        """Generate a single LCM task"""
        rng = Random(self.seed + idx)
        
        numbers = self._generate_numbers(rng)
        result = self._calculate_lcm(numbers)
        
        numbers_str = ", ".join(str(n) for n in numbers)
        
        return {
            "question": f"Find the Least Common Multiple (LCM) of these numbers: {numbers_str}",
            "answer": str(result),
            "metadata": {
                "numbers": numbers,
                "result": result
            }
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
