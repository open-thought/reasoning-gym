"""Prime factorization task generator"""
from dataclasses import dataclass
from random import Random
from typing import List, Optional, Tuple

@dataclass
class PrimeFactorizationConfig:
    """Configuration for prime factorization task generation"""
    min_value: int = 2         # Minimum number to factorize
    max_value: int = 1000      # Maximum number to factorize
    seed: Optional[int] = None
    size: int = 500           # Virtual dataset size

    def validate(self):
        """Validate configuration parameters"""
        assert self.min_value >= 2, "min_value must be >= 2"
        assert self.max_value >= self.min_value, "max_value must be >= min_value"


class PrimeFactorizationDataset:
    """Generates prime factorization tasks"""

    def __init__(self, config: PrimeFactorizationConfig):
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

    def _prime_factors(self, n: int) -> List[int]:
        """Compute prime factors of a number"""
        factors = []
        d = 2
        while n > 1:
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1
            if d * d > n:
                if n > 1:
                    factors.append(n)
                break
        return factors

    def __getitem__(self, idx: int) -> dict:
        """Generate a single prime factorization task"""
        rng = Random(self.seed + idx)
        
        # Generate random number to factorize
        number = rng.randint(self.config.min_value, self.config.max_value)
        
        # Calculate prime factors
        factors = self._prime_factors(number)
        
        # Format answer as multiplication of prime factors
        answer = " × ".join(map(str, factors))
        
        return {
            "question": (f"Find the prime factorization of {number}. "
                       f"(Example: 12 = 2 × 2 × 3)"),
            "answer": answer,
            "metadata": {
                "number": number,
                "factors": factors
            }
        }


def prime_factorization_dataset(
    min_value: int = 2,
    max_value: int = 1000,
    seed: Optional[int] = None,
    size: int = 500,
) -> PrimeFactorizationDataset:
    """Create a PrimeFactorizationDataset with the given configuration."""
    config = PrimeFactorizationConfig(
        min_value=min_value,
        max_value=max_value,
        seed=seed,
        size=size,
    )
    return PrimeFactorizationDataset(config)
