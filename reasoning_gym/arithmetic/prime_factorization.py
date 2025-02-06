"""Prime factorization task generator"""

from dataclasses import dataclass
from random import Random
from typing import List, Optional

from ..factory import ProceduralDataset, register_dataset


@dataclass
class PrimeFactorizationConfig:
    """Configuration for prime factorization task generation"""

    min_value: int = 2  # Minimum number to factorize
    max_value: int = 1000  # Maximum number to factorize
    seed: Optional[int] = None
    size: int = 500  # Virtual dataset size

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.min_value >= 2, "min_value must be >= 2"
        assert self.max_value >= self.min_value, "max_value must be >= min_value"


class PrimeFactorizationDataset(ProceduralDataset):
    """Generates prime factorization tasks"""

    def __init__(self, config: PrimeFactorizationConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

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
            "question": (
                f"Find the prime factorization of {number}. Write the factors separated by × "
                f"(Example: for 12 the answer would be: 2 × 2 × 3)"
            ),
            "answer": answer,
            "metadata": {"number": number, "factors": factors},
        }


register_dataset("prime_factorization", PrimeFactorizationDataset, PrimeFactorizationConfig)
