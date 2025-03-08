"""Computhe the power of a number."""

from dataclasses import dataclass
from decimal import Decimal
from math import pow
from random import Random
from typing import Any, Optional

from ..coaching import AttributeType, BaseCurriculum, RangeAttributeDefinition
from ..factory import ProceduralDataset, register_dataset

QUESTION_TEMPLATE = """Your task is to compute an exponentiation of a number.

Compute {base}^{exponent}. Return your final answer correct to 3 significant figures.
Provide your answer in scientific notation using 'e' notation (e.g., 1.23e+4).
"""


@dataclass
class PowerFunctionConfig:
    """Configuration for Power Function dataset generation"""

    min_base: float = -1e3  # Minimum base value
    max_base: float = 1e3  # Maximum base value
    min_exponent: int = 0  # Minimum exponent value
    max_exponent: int = 8  # Maximum exponent value

    size: int = 500  # Virtual dataset size
    seed: Optional[int] = None


class PowerFunctionDataset(ProceduralDataset):
    """Generates Power Function exercises with configurable difficulty"""

    def __init__(self, config: PowerFunctionConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """Score the answer by checking if it matches the expected answer to 3 significant figures."""
        oracle_answer = entry["answer"]
        if answer is not None:
            try:
                user_answer = Decimal(answer)
                oracle_value = Decimal(oracle_answer)

                if oracle_value == 0:
                    return 1.0 if user_answer == 0 else 0.01

                user_sig_figs = f"{user_answer:.3g}"
                oracle_sig_figs = f"{oracle_value:.3g}"

                # Check if they match to 3 significant figures
                if user_sig_figs == oracle_sig_figs:
                    return 1.0
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

        if rng.random() < 0.5:
            exponent = -exponent

        answer = pow(base, exponent)

        return {
            "question": QUESTION_TEMPLATE.format(base=base, exponent=exponent),
            "answer": str(answer),
            "metadata": {"base": base, "exponent": exponent, "solution": answer, "difficulty": {"exponent": exponent}},
        }


class PowerFunctionCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(PowerFunctionCurriculum.__name__, PowerFunctionConfig)
        self._define_attributes(
            RangeAttributeDefinition(
                name="exponent",
                levels=[2, 4, 6, 10],
                default_level=0,
                attr_type=AttributeType.APPEND,
                min_value=2,
                lower_field_name="min_exponent",
                upper_field_name="max_exponent",
            ),
        )


register_dataset("power_function", PowerFunctionDataset, PowerFunctionConfig, PowerFunctionCurriculum)
