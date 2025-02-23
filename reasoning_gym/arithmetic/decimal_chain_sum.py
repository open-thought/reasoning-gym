import random
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Optional

from ..coaching import AttributeType, BaseCurriculum, RangeAttributeDefinition
from ..factory import ProceduralDataset, register_dataset


@dataclass
class DecimalChainSumConfig:
    """Configuration for decimal chain sum task generation"""

    min_terms: int = 2
    max_terms: int = 6
    min_digits: int = 1
    max_digits: int = 4
    min_decimal_places: int = 1
    max_decimal_places: int = 4
    allow_negation: bool = False
    seed: Optional[int] = None
    size: int = 500

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.size > 0, "size must be positive"
        assert self.min_terms > 0, "min_terms must be positive"
        assert self.max_terms >= self.min_terms, "max_terms must be >= min_terms"
        assert self.min_digits > 0, "min_digits must be positive"
        assert self.max_digits >= self.min_digits, "max_digits must be >= min_digits"
        assert self.min_decimal_places >= 0, "min_decimal_places must be non-negative"
        assert self.max_decimal_places >= self.min_decimal_places, "max_decimal_places must be >= min_decimal_places"


class DecimalChainSumDataset(ProceduralDataset):
    """Generates simple decimal arithmetic tasks using only + and - operators"""

    def __init__(self, config: DecimalChainSumConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def __getitem__(self, idx: int) -> dict:
        """Generate a single decimal chain sum task

        Args:
            idx: Index of the item to generate

        Returns:
            dict with keys:
                - question: str, the formatted arithmetic expression
                - answer: str, the ground truth result
                - metadata: dict with generation parameters
        """

        rng = random.Random(self.seed + idx)

        num_terms = rng.randint(self.config.min_terms, self.config.max_terms)
        num_digits = rng.randint(self.config.min_digits, self.config.max_digits)

        # Calculate value ranges based on number of digits
        min_value = 0 if num_digits == 1 else 10 ** (num_digits - 1)  # Special case for 1 digit
        max_value = (10**num_digits) - 1  # e.g., 999 for 3 digits

        expression, result = self._generate_task(rng, num_terms, min_value, max_value)

        return {
            "question": f"State the final answer to the following arithmetic problem: {expression} =",
            "answer": str(result),
            "metadata": {
                "difficulty": {
                    "num_terms": num_terms,
                    "num_digits": num_digits,
                },
                "expression": expression,
            },
        }

    def _generate_task(self, rng: random.Random, num_terms: int, min_value: int, max_value: int) -> tuple[str, Decimal]:
        """Generate a single decimal chain sum task

        Args:
            rng: Random number generator
            num_terms: Number of terms in the expression
            min_value: Minimum value for generated numbers
            max_value: Maximum value for generated numbers
            min_decimal_places: Minimum number of decimal places
            max_decimal_places: Maximum number of decimal places

        Returns:
            Tuple of (expression string, result Decimal)
        """

        # Convert constants to Decimal
        constants = [
            Decimal(
                str(
                    rng.randint(-max_value, max_value)
                    if self.config.allow_negation
                    else rng.randint(min_value, max_value)
                )
            )
            for _ in range(num_terms)
        ]

        # Generate decimal places for each term
        decimal_places = [
            rng.randint(self.config.min_decimal_places, self.config.max_decimal_places) for _ in range(num_terms)
        ]

        # Add decimal parts using Decimal for precise arithmetic
        for i in range(num_terms):
            min_val = 0 if decimal_places[i] == 0 else 10 ** (decimal_places[i] - 1)
            max_val = (10 ** decimal_places[i]) - 1
            decimal_part = Decimal(str(rng.randint(min_val, max_val))) / Decimal(str(10 ** decimal_places[i]))
            constants[i] += decimal_part

        operators = [rng.choice(["+", "-"]) for _ in range(num_terms - 1)]

        expression_parts = []
        result = constants[0]

        expression_parts.append(f"{constants[0]:.{decimal_places[0]}f}")
        for i, op in enumerate(operators):
            c = constants[i + 1]
            expression_parts.append(op)
            expression_parts.append(f"{c:.{decimal_places[i+1]}f}")

            if op == "+":
                result += c
            else:  # op == "-"
                result -= c

        expression = " ".join(expression_parts)
        result = result.quantize(Decimal(f"0.{'0' * max(decimal_places)}"))
        return expression, result

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """Score the answer by comparing decimal values instead of strings.
        Args:
            answer: The answer to score
            entry: The entry containing the oracle answer

        Returns:
            1.0 for exact numerical match, 0.01 otherwise
        """
        if answer is None or len(answer.strip()) == 0:
            return 0.0

        try:
            student_answer = Decimal(answer.strip())
            oracle_answer = Decimal(entry["answer"])

            return 1.0 if student_answer == oracle_answer else 0.01
        except (ValueError, TypeError, ArithmeticError):
            return 0.01


register_dataset("decimal_chain_sum", DecimalChainSumDataset, DecimalChainSumConfig)
