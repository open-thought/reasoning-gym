import math
import random
from dataclasses import dataclass
from fractions import Fraction
from typing import Optional

from reasoning_gym.dataset import ProceduralDataset

from ..coaching import BaseCurriculum, RangeAttributeDefinition
from ..factory import register_dataset

DATASET_NAME = "coin_flip"


@dataclass
class CoinFlipConfig:
    """Configuration for coin flip probability task generation."""

    min_trials: int = 3
    max_trials: int = 15
    allow_exact: bool = True  # whether to allow "exactly k heads" problems
    allow_at_least: bool = True  # whether to allow "at least k heads" problems
    seed: Optional[int] = None
    size: int = 500

    def validate(self) -> None:
        assert self.size > 0, "size must be positive"
        assert self.min_trials > 0, "min_trials must be positive"
        assert self.max_trials >= self.min_trials, "max_trials must be >= min_trials"
        assert self.allow_exact or self.allow_at_least, "At least one of allow_exact or allow_at_least must be True"


class CoinFlipDataset(ProceduralDataset):
    """Generates coin-flip probability problems (exact k heads / at-least k heads)."""

    def __init__(self, config: CoinFlipConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def __getitem__(self, idx: int) -> dict:
        """
        Generate a single N coin flip probability problem.
        Args:
            idx: Index of the item to generate

        Returns:
            dict with keys:
                - question: str, the formatted arithmetic expression
                - answer: str, the ground truth result
                - metadata: dict with generation parameters
        """
        # Create deterministic RNG from base seed and idx
        rng = random.Random(self.seed + idx)

        # Pick number of trials
        n = rng.randint(self.config.min_trials, self.config.max_trials)

        available_types = []
        if self.config.allow_exact:
            available_types.append("exact")
        if self.config.allow_at_least:
            available_types.append("at_least")

        problem_type = rng.choice(available_types)

        if problem_type == "exact":
            k = rng.randint(0, n)
            question = f"What is the probability of getting exactly {k} heads in {n} fair coin flips?"
            prob = self._prob_exact_heads(n, k)  # compute actual answer as float

        else:
            k = rng.randint(0, n)
            question = f"What is the probability of getting at least {k} heads in {n} fair coin flips?"
            prob = self._prob_at_least_heads(n, k)  # compute actual answer as float

        answer_str = format(prob, ".10g")

        return {
            "question": question,
            "answer": answer_str,
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "num_trials": n,
                "k_heads": k,
                "problem_type": problem_type,
                "rational": {
                    "numerator": self._rational_numerator(n, k, problem_type),
                    "denominator": 2**n,
                },
                "difficulty": {
                    "num_trials": (self.config.min_trials, self.config.max_trials),
                },
            },
        }

    def _prob_exact_heads(self, n: int, k: int) -> float:
        """Return probability of exactly k heads in n fair coin tosses."""
        comb = math.comb(n, k)
        return comb * (0.5**n)

    def _prob_at_least_heads(self, n: int, k: int) -> float:
        """Return probability of at least k heads in n fair coin tosses."""
        total = sum(math.comb(n, i) for i in range(k, n + 1))
        return total * (0.5**n)

    def _rational_numerator(self, n: int, k: int, problem_type: str) -> int:
        """Return the numerator of the probability as a rational number."""
        if problem_type == "exact":
            return math.comb(n, k)
        else:
            return sum(math.comb(n, i) for i in range(k, n + 1))

    def score_answer(self, answer: Optional[str], entry: dict, tol: float = 1e-4) -> float:
        """
        Compute reward for LLM answer against oracle probability.
        Handles decimals, fractions, small numeric errors, and extra text.
        """
        reward = 0.0
        oracle_answer = entry["answer"]

        if answer is None or len(answer.strip()) == 0:
            return reward

        answer = answer.replace(",", "")
        oracle_answer = oracle_answer.replace(",", "")

        try:
            answer_float = float(Fraction(answer))
            oracle_answer_float = float(Fraction(oracle_answer))
        except (ValueError, ZeroDivisionError):
            return reward

        if abs(answer_float - oracle_answer_float) <= tol:
            return 1.0

        answer_str = f"{answer_float:.10g}"
        oracle_answer_str = f"{oracle_answer_float:.10g}"

        # Partial Reward for matching prefix
        match_len = 0
        for a_char, o_char in zip(answer_str, oracle_answer_str):
            if a_char == o_char:
                match_len += 1
            else:
                break

        reward = match_len / min(len(oracle_answer_str), len(answer_str))

        return reward


class CoinFlipCurriculum(BaseCurriculum):
    """Curriculum that allows scaling the number of tosses."""

    def __init__(self):
        super().__init__(CoinFlipCurriculum.__name__, CoinFlipConfig)
        self._define_attributes(
            RangeAttributeDefinition(
                name="num_trials",
                levels=list(range(3, 16)),  # starting from 3 upto 15 tosses
                default_level=0,
                description="Number of coin tosses (difficulty)",
                lower_field_name="min_trials",
                upper_field_name="max_trials",
            ),
        )


register_dataset(DATASET_NAME, CoinFlipDataset, CoinFlipConfig, CoinFlipCurriculum)
