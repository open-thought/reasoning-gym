"""Computhe the power of a number."""

from dataclasses import dataclass
from math import pow
from random import Random
from typing import Dict, Optional

from ..factory import ProceduralDataset, register_dataset

QUESTION_TEMPLATE = """Compute {base}^{exponent}"""


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

    def score_answer(self, answer: Optional[str], entry: Dict[str, any]) -> float:
        """Overwrite this method in derived classes if a single oracle answer is not available."""
        oracle_answer = entry["answer"]
        reward = 0.0
        if answer is not None:
            difference = abs(float(answer) - float(oracle_answer))
            if difference < 1e-6:
                reward = 1.0
            elif difference < 1e-1:
                reward = 0.5
            else:
                reward = 0.01

        return reward

    def __getitem__(self, idx: int) -> dict:
        """Generate a single Power Function question"""
        rng = Random(self.seed + idx)

        base = rng.uniform(self.config.min_base, self.config.max_base)
        exponent = rng.randint(self.config.min_exponent, self.config.max_exponent)
        answer = pow(base, exponent)

        return {
            "question": f"Compute {base}^{exponent}",
            "answer": str(answer),
            "metadata": {"base": base, "exponent": exponent, "solution": answer},
        }


register_dataset("power_function", PowerFunctionDataset, PowerFunctionConfig)
