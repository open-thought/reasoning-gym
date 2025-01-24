"""Greatest Common Divisor (GCD) task generator"""
from dataclasses import dataclass
from random import Random
from typing import List, Optional, Tuple
from math import gcd
from functools import reduce


@dataclass
class GCDConfig:
    """Configuration for GCD task generation"""
    min_numbers: int = 2       # Minimum numbers to find GCD of
    max_numbers: int = 2       # Maximum numbers to find GCD of
    min_value: int = 1        # Minimum value for each number
    max_value: int = 1000     # Maximum value for each number
    seed: Optional[int] = None
    size: int = 500          # Virtual dataset size

    def validate(self):
        """Validate configuration parameters"""
        assert self.min_numbers >= 2, "min_numbers must be at least 2"
        assert self.max_numbers >= self.min_numbers, "max_numbers must be >= min_numbers"
        assert self.min_value >= 1, "min_value must be positive"
        assert self.max_value > self.min_value, "max_value must be > min_value"


class GCDDataset:
    """Generates Greatest Common Divisor (GCD) tasks"""

    def __init__(self, config: GCDConfig):
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

    def _calculate_gcd(self, numbers: List[int]) -> int:
        """Calculate the GCD of a list of numbers"""
        return reduce(gcd, numbers)

    def __getitem__(self, idx: int) -> dict:
        """Generate a single GCD task"""
        rng = Random(self.seed + idx)
        
        numbers = self._generate_numbers(rng)
        result = self._calculate_gcd(numbers)
        
        numbers_str = ", ".join(str(n) for n in numbers)
        
        return {
            "question": f"Find the Greatest Common Divisor (GCD) of these numbers: {numbers_str}",
            "answer": str(result),
            "metadata": {
                "numbers": numbers,
                "result": result
            }
        }


def gcd_dataset(
    min_numbers: int = 2,
    max_numbers: int = 2,
    min_value: int = 1,
    max_value: int = 1000,
    seed: Optional[int] = None,
    size: int = 500,
) -> GCDDataset:
    """Create a GCDDataset with the given configuration."""
    config = GCDConfig(
        min_numbers=min_numbers,
        max_numbers=max_numbers,
        min_value=min_value,
        max_value=max_value,
        seed=seed,
        size=size,
    )
    return GCDDataset(config)
