"""Choose largest number out of several represented in various formats."""

from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from ..factory import ProceduralDataset, register_dataset

QUESTION_TEMPLATE = """Your task is to pick the largest/smallest number out of several options.

Example
- Input: Pick the largest number of the following candidates: 857575.23 8.975554e+05 887,555.62
- Output: 8.975554e+05
- Explanation:
    - Sorting the numbers written in various notations we get: 857575.23 < 887,555.62 < 8.975554e+05
    - Therefore, the largest number is 8.975554e+05

Now, pick the {size} number of the following candidates: {numbers}
"""


@dataclass
class NumberFormatConfig:
    """Configuration for Count Bits dataset generation"""

    max_num_candidates: int = 5  # Maximum number of candidates
    min_n: float = 1_000  # Lower bound for the numbers
    max_n: float = 1_000_000_000  # Upper bound for the numbers
    max_delta: int = 1_000

    size: int = 500  # Virtual dataset size
    seed: Optional[int] = None

    def validate(self):
        """Validate configuration parameters"""
        assert 2 <= self.max_num_candidates, "max_num_candidates must be at least 2"
        assert 1 <= self.min_n, "min_n must be at least 1"
        assert self.min_n < self.max_n, "min_n must be less than max_n"
        assert 1 <= self.max_delta, "max_delta must be at least 1"


class NumberFormatDataset(ProceduralDataset):
    """Generates Count Bits exercises with configurable difficulty"""

    def __init__(self, config: NumberFormatConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _get_candidates(self, rng: Random, num_candidates: int) -> list:
        """Generate a list of candidates"""
        base = round(rng.uniform(self.config.min_n, self.config.max_n), 3)
        candidates = [base]
        for _ in range(num_candidates - 1):
            delta = round(rng.uniform(-self.config.max_delta, self.config.max_delta), 3)
            candidates.append(base + delta)
        return candidates

    def _transform_candidates(self, rng: Random, candidates: list[float]) -> list[str]:
        """Randomly apply different number formats to the candidates"""
        output = []
        for candidate in candidates:
            format_type = rng.choice(["standard", "english", "scientific"])
            if format_type == "standard":
                output.append(f"{candidate:f}")
            elif format_type == "english":
                output.append(f"{candidate:,}")
            elif format_type == "scientific":
                output.append(f"{candidate:.15e}")
        return output

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """Overwrite this method in derived classes if a single oracle answer is not available."""
        oracle_answer = entry["metadata"]["solution"]
        if answer is not None and len(answer) > 0:
            try:
                answer = float(answer.strip().replace(",", ""))
                if abs(answer - oracle_answer) < 1e-2:
                    return 1.0
                return 0.01
            except:
                return 0.0
        return 0.0

    def __getitem__(self, idx: int) -> dict:
        """Generate a single Count Bits question"""
        rng = Random(self.seed + idx)

        num_candidates = rng.randint(2, self.config.max_num_candidates)
        candidates = self._get_candidates(rng, num_candidates)
        formatted_candidates = self._transform_candidates(rng, candidates)

        size = rng.choice(["largest", "smallest"])
        answer = max(candidates) if size == "largest" else min(candidates)

        return {
            "question": QUESTION_TEMPLATE.format(numbers=" ".join(formatted_candidates), size=size),
            "answer": str(answer),
            "metadata": {
                "candidates": candidates,
                "solution": answer,
                "formatted_candidates": formatted_candidates,
                "size": size,
            },
        }


register_dataset("number_format", NumberFormatDataset, NumberFormatConfig)
