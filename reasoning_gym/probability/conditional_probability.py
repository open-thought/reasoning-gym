import random
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Optional

from ..coaching import BaseCurriculum, RangeAttributeDefinition, ScalarAttributeDefinition
from ..factory import ProceduralDataset, register_dataset

DATASET_NAME = "conditional_probability"

TASK_TYPES = ("bayes", "dependent_draws", "contingency_table")


@dataclass
class ConditionalProbabilityConfig:
    task_types: tuple[str, ...] = TASK_TYPES
    task_weights: list[float] = field(default_factory=lambda: [0.34, 0.33, 0.33])
    min_total_items: int = 5
    max_total_items: int = 20
    min_table_cell: int = 5
    max_table_cell: int = 50
    seed: Optional[int] = None
    size: int = 500

    def validate(self) -> None:
        assert self.size > 0, "size must be positive"
        assert len(self.task_types) > 0, "must have at least one task type"
        assert all(t in TASK_TYPES for t in self.task_types), f"invalid task type"
        assert len(self.task_weights) == len(self.task_types), "weights must match types"
        assert self.min_total_items >= 2, "min_total_items must be >= 2"
        assert self.max_total_items >= self.min_total_items, "max_total_items must be >= min_total_items"


class ConditionalProbabilityDataset(ProceduralDataset):
    def __init__(self, config: ConditionalProbabilityConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _make_bayes(self, rng: random.Random) -> dict:
        sensitivity = Fraction(rng.randint(70, 99), 100)
        specificity = Fraction(rng.randint(70, 99), 100)
        prevalence = Fraction(rng.randint(1, 15), 100)

        p_pos = sensitivity * prevalence + (1 - specificity) * (1 - prevalence)
        p_disease_given_pos = (sensitivity * prevalence) / p_pos

        question = (
            f"A medical test has a sensitivity (true positive rate) of {sensitivity} "
            f"and a specificity (true negative rate) of {specificity}. "
            f"The prevalence of the disease in the population is {prevalence}. "
            f"If a person tests positive, what is the probability they actually have the disease? "
            f"Give your answer as a simplified fraction."
        )
        return {"question": question, "answer": str(p_disease_given_pos), "task_type": "bayes"}

    def _make_dependent_draws(self, rng: random.Random) -> dict:
        total = rng.randint(self.config.min_total_items, self.config.max_total_items)
        color_a_count = rng.randint(2, total - 1)
        color_b_count = total - color_a_count
        draws = rng.randint(2, min(3, color_a_count))

        color_a = rng.choice(["red", "blue", "green", "white", "black"])
        color_b = rng.choice([c for c in ["red", "blue", "green", "white", "black"] if c != color_a])

        prob = Fraction(1, 1)
        for i in range(draws):
            prob *= Fraction(color_a_count - i, total - i)

        question = (
            f"A bag contains {color_a_count} {color_a} balls and {color_b_count} {color_b} balls. "
            f"You draw {draws} balls without replacement. "
            f"What is the probability that all {draws} balls are {color_a}? "
            f"Give your answer as a simplified fraction."
        )
        return {"question": question, "answer": str(prob), "task_type": "dependent_draws"}

    def _make_contingency(self, rng: random.Random) -> dict:
        a = rng.randint(self.config.min_table_cell, self.config.max_table_cell)
        b = rng.randint(self.config.min_table_cell, self.config.max_table_cell)
        c = rng.randint(self.config.min_table_cell, self.config.max_table_cell)
        d = rng.randint(self.config.min_table_cell, self.config.max_table_cell)

        total = a + b + c + d
        row1_total = a + b
        prob = Fraction(a, row1_total)

        question = (
            f"Consider the following contingency table:\n\n"
            f"              | Event B | Not B | Total\n"
            f"  Event A     |   {a:>4}  | {b:>4} | {row1_total:>4}\n"
            f"  Not A       |   {c:>4}  | {d:>4} | {c + d:>4}\n"
            f"  Total       |   {a + c:>4}  | {b + d:>4} | {total:>4}\n\n"
            f"Given that Event A occurred, what is the probability of Event B? "
            f"Give your answer as a simplified fraction."
        )
        return {"question": question, "answer": str(prob), "task_type": "contingency_table"}

    def __getitem__(self, idx: int) -> dict:
        rng = random.Random(self.seed + idx)
        task_type = rng.choices(self.config.task_types, weights=self.config.task_weights, k=1)[0]

        generators = {
            "bayes": self._make_bayes,
            "dependent_draws": self._make_dependent_draws,
            "contingency_table": self._make_contingency,
        }
        result = generators[task_type](rng)
        return {
            "question": result["question"],
            "answer": result["answer"],
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "task_type": result["task_type"],
                "difficulty": {
                    "min_total_items": self.config.min_total_items,
                    "max_total_items": self.config.max_total_items,
                },
            },
        }

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        if answer is None:
            return 0.0
        oracle = entry["answer"]
        if answer.strip() == oracle.strip():
            return 1.0
        try:
            ans_frac = Fraction(answer.strip())
            oracle_frac = Fraction(oracle.strip())
            if ans_frac == oracle_frac:
                return 1.0
            diff = abs(float(ans_frac) - float(oracle_frac))
            if diff < 1e-4:
                return 0.9
            if diff < 1e-2:
                return 0.5
            return 0.0
        except (ValueError, ZeroDivisionError):
            return 0.0


class ConditionalProbabilityCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(ConditionalProbabilityCurriculum.__name__, ConditionalProbabilityConfig)
        self._define_attributes(
            RangeAttributeDefinition(
                name="total_items",
                levels=[5, 10, 20, 50],
                lower_field_name="min_total_items",
                upper_field_name="max_total_items",
                description="Total items for draw problems",
            ),
        )


register_dataset(
    DATASET_NAME, ConditionalProbabilityDataset, ConditionalProbabilityConfig, ConditionalProbabilityCurriculum
)
