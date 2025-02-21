"""Computhe the power of a number."""

from dataclasses import dataclass
from math import pow
from random import Random
from typing import Any, Optional

from ..factory import ProceduralDataset, register_dataset

QUESTION_TEMPLATE = """Your task is to compute an exponentiation of a number.

Example:
- Input: Compute 2^3
- Output: 8
- Explanation:
    - 2^3 = 2 * 2 * 2 = 8
    - Therefore, the final answer is 8

Example:
- Input: Compute 412.5^3
- Output: 70189453.125
- Explanation:
    - 412.5^3 = 412.5 * 412.5 * 412.5 = 70189453.125
    - Therefore, the final answer is 70189453.125

Compute {base}^{exponent}
"""


@dataclass
class PowerFunctionConfig:
    """Configuration for Power Function dataset generation"""

    min_base: float = -1e3  # Minimum base value
    max_base: float = 1e3  # Maximum base value
    min_exponent: int = -8  # Minimum exponent value
    max_exponent: int = 8  # Maximum exponent value

    size: int = 500  # Virtual dataset size
    seed: Optional[int] = None


class PowerFunctionDataset(ProceduralDataset):
    """Generates Power Function exercises with configurable difficulty"""

    def __init__(self, config: PowerFunctionConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """Overwrite this method in derived classes if a single oracle answer is not available."""
        oracle_answer = entry["answer"]
        if answer is not None:
            try:
                answer = round(float(answer), 4)
                oracle_answer = round(float(oracle_answer), 4)
                difference = abs(float(answer) - float(oracle_answer))
                if difference < 1e-4:
                    return 1.0
                elif difference < 1e-1:
                    return 0.5
                else:
                    return 0.01
            except Exception as e:
                return 0.01
        return 0.0

    def __getitem__(self, idx: int) -> dict:
        """Generate a single Power Function question"""
        rng = Random(self.seed + idx)

        base = round(rng.uniform(self.config.min_base, self.config.max_base), 4)
        exponent = rng.randint(self.config.min_exponent, self.config.max_exponent)
        answer = pow(base, exponent)

        return {
            "question": QUESTION_TEMPLATE.format(base=base, exponent=exponent),
            "answer": str(answer),
            "metadata": {"base": base, "exponent": exponent, "solution": answer},
        }


register_dataset("power_function", PowerFunctionDataset, PowerFunctionConfig)
