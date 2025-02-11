"""Count prime numbers in a given interval.

Solution obtained with Sieve of Eratosthenes:
https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes
"""

import math
from dataclasses import dataclass
from random import Random
from typing import Optional

from ..factory import ProceduralDataset, register_dataset

QUESTION_TEMPLATE = """Count how many prime numbers there are between {start} and {end} (inclusive) ?"""


@dataclass
class CountPrimesConfig:
    """Configuration for Count Primes dataset generation"""

    max_n: int = 10_000  # Upper bound for the interval

    size: int = 500  # Virtual dataset size
    seed: Optional[int] = None

    def validate(self):
        """Validate configuration parameters"""
        assert 1 <= self.max_n, "max_n must be at least 1"


class CountPrimesDataset(ProceduralDataset):
    """Generates Count Primes exercises with configurable difficulty"""

    def __init__(self, config: CountPrimesConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)
        self.primes = self._get_primes(config.max_n + 1)

    def _get_primes(self, n: int) -> list[bool]:
        if n <= 1:
            return []
        primes = [True] * n
        primes[0] = primes[1] = False
        for i in range(2, int(math.sqrt(n)) + 1):
            if primes[i]:
                for j in range(2 * i, n, i):
                    primes[j] = False
        return primes

    def __getitem__(self, idx: int) -> dict:
        """Generate a single Count Primes question"""
        rng = Random(self.seed + idx)
        start = rng.randint(1, self.config.max_n)
        end = rng.randint(start, self.config.max_n)
        primes = self.primes[start : end + 1]
        answer = sum(primes)
        return {
            "question": QUESTION_TEMPLATE.format(start=start, end=end),
            "answer": str(answer),
            "metadata": {"start": start, "end": end, "primes": primes, "solution": answer},
        }


register_dataset("count_primes", CountPrimesDataset, CountPrimesConfig)
