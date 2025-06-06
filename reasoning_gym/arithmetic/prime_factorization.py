"""Prime factorization task generator"""

import math
from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from ..coaching import BaseCurriculum, RangeAttributeDefinition
from ..factory import ProceduralDataset, register_dataset

DATASET_NAME = "prime_factorization"


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

    def _prime_factors(self, n: int) -> list[int]:
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

    def _normalize_answer(self, answer: str) -> list[int]:
        """Parse and sort factors from a string"""
        if not answer or answer.strip() == "":
            return []

        return sorted([int(factor.strip()) for factor in answer.split("×") if factor.strip() != ""])

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        oracle_answer = entry["answer"]
        reward = 0.0
        if answer is not None:
            oracle_answer_parsed = self._normalize_answer(oracle_answer)
            answer_parsed = self._normalize_answer(answer)
            if oracle_answer_parsed == answer_parsed:
                reward = 1.0
            elif math.prod(oracle_answer_parsed) == math.prod(answer_parsed):
                reward = 0.5
            else:
                reward = 0.01

        return reward

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
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "number": number,
                "factors": factors,
                "difficulty": {
                    "value": (self.config.min_value, self.config.max_value),
                },
            },
        }


class PrimeFactorizationCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(PrimeFactorizationCurriculum.__name__, PrimeFactorizationConfig)

        # Define attributes
        self._define_attributes(
            RangeAttributeDefinition(
                name="value",
                levels=[10, 1_000, 5_000, 10_000],
                description="Number to factorize",
                lower_field_name="min_value",
                upper_field_name="max_value",
                ensure_interval=True,
            )
        )


register_dataset(DATASET_NAME, PrimeFactorizationDataset, PrimeFactorizationConfig, PrimeFactorizationCurriculum)
