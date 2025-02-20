"""Comparing two decimal numbers"""

from dataclasses import dataclass
from decimal import Decimal
from random import Random
from typing import Optional

from ..factory import ProceduralDataset, register_dataset


@dataclass
class DecimalNumberComparisonConfig:
    """Configuration for number comparison task generation"""

    min_precision: int = 5
    max_precision: int = 20
    seed: Optional[int] = None
    min_number: int = 3
    max_number: int = 10
    size: int = 500

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.min_precision >= 5, "min_precision must be greater than or equal to 5"
        assert self.min_precision <= self.max_precision, "min_precision must be <= max_precision"
        assert self.max_precision >= self.min_precision, "max_precision must be >= min_precision"
        assert self.min_number >= 3, "min_number must be greater than or equal to 3"
        assert self.min_number <= self.max_number, "min_number must be <= max_number"
        assert self.max_number >= self.min_number, "max_number must be <= min_precision"


class DecimalNumberComparisonDataset(ProceduralDataset):
    """Generates sentence reordering tasks from text spans"""

    def __init__(self, config: DecimalNumberComparisonConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def compare_decimals(self, a: Decimal, b: Decimal) -> str:
        """Compare two decimal numbers"""
        if a < b:
            return "<"
        elif a == b:
            return "="
        return ">"

    def __getitem__(self, idx: int) -> dict:
        """Generate a single entity-based number comparison task"""
        rng = Random(self.seed + idx)

        # Randomly choose precision for each decimal (between 1 and 6 decimal places)
        precision_1 = rng.randint(self.config.min_precision, self.config.max_precision)
        precision_2 = rng.randint(self.config.min_precision, self.config.max_precision)

        # Generate two random floats with different precision
        num1 = round(rng.uniform(self.config.min_number, self.config.max_number), precision_1)
        num2 = round(rng.uniform(self.config.min_number, self.config.max_number), precision_2)
        dec1 = Decimal(str(num1))
        dec2 = Decimal(str(num2))
        question = f"Is {dec1} >, < or = {dec2}? Answer with either of the 3 symbols."
        answer = self.compare_decimals(dec1, dec2)

        return {
            "question": question,
            "answer": answer,
            "metadata": {
                "min_precision": min(precision_1, precision_2),
                "max_precision": max(precision_1, precision_2),
                "min_number": min(num1, num2),
                "max_number": max(num1, num2),
            },
        }


register_dataset("decimal_number_comparison", DecimalNumberComparisonDataset, DecimalNumberComparisonConfig)
