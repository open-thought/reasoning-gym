import random
from dataclasses import dataclass
from typing import Optional

from ..factory import ProceduralDataset, register_dataset


@dataclass
class ChainSumConfig:
    """Configuration for chain sum task generation"""

    min_terms: int = 2
    max_terms: int = 6
    min_digits: int = 1
    max_digits: int = 4
    allow_negation: bool = False
    seed: Optional[int] = None
    size: int = 500

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.size > 0, "size must be positive"
        """Validate configuration parameters"""
        assert self.min_terms > 0, "min_terms must be positive"
        assert self.max_terms >= self.min_terms, "max_terms must be >= min_terms"
        assert self.min_digits > 0, "min_digits must be positive"
        assert self.max_digits >= self.min_digits, "max_digits must be >= min_digits"

        # Validate digit ranges make sense
        if self.min_digits > 1:
            assert 10 ** (self.min_digits - 1) >= 1, "min_digits would result in invalid number range"


class ChainSum(ProceduralDataset):
    """Generates simple arithmetic tasks using only + and - operators"""

    def __init__(self, config: ChainSumConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def __getitem__(self, idx: int) -> dict:
        """Generate a single chain sum task

        Args:
            idx: Index of the item to generate

        Returns:
            dict with keys:
                - question: str, the formatted arithmetic expression
                - answer: str, the ground truth result
                - metadata: dict with generation parameters
        """
        # Create deterministic RNG from base seed and idx
        item_rng = random.Random(self.seed + idx)

        num_terms = item_rng.randint(self.config.min_terms, self.config.max_terms)
        num_digits = item_rng.randint(self.config.min_digits, self.config.max_digits)

        # Calculate value ranges based on number of digits
        min_value = 0 if num_digits == 1 else 10 ** (num_digits - 1)  # Special case for 1 digit
        max_value = (10**num_digits) - 1  # e.g., 999 for 3 digits

        expression, result = self._generate_task(item_rng, num_terms, min_value, max_value)

        return {
            "question": f"{expression} =",
            "answer": str(result),
            "metadata": {
                "difficulty": {
                    "num_terms": num_terms,
                    "num_digits": num_digits,
                },
                "expression": expression,
            },
        }

    def _generate_task(self, rng: random.Random, num_terms: int, min_value: int, max_value: int) -> tuple[str, int]:
        """Generate a chain sum task

        Args:
            rng: Random number generator
            num_terms: Number of terms in the expression
            min_value: Minimum value for generated numbers
            max_value: Maximum value for generated numbers

        Returns:
            Tuple of (expression string, result integer)
        """
        if self.config.allow_negation:
            # Allow both positive and negative numbers in the range
            constants = [rng.randint(-max_value, max_value) for _ in range(num_terms)]
        else:
            # Only positive numbers
            constants = [rng.randint(min_value, max_value) for _ in range(num_terms)]
        operators = [rng.choice(["+", "-"]) for _ in range(num_terms - 1)]

        # Build expression and compute result
        expression_parts = []
        result = constants[0]

        expression_parts.append(str(constants[0]))
        for i, op in enumerate(operators):
            c = constants[i + 1]
            expression_parts.append(op)
            expression_parts.append(str(c))

            if op == "+":
                result += c
            else:  # op == "-"
                result -= c

        expression = " ".join(expression_parts)
        return expression, result


# Register the dataset
register_dataset("chain_sum", ChainSum, ChainSumConfig)
