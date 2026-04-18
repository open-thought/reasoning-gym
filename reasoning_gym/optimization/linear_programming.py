import random
from dataclasses import dataclass, field
from typing import Any, Optional

from ..coaching import BaseCurriculum, ScalarAttributeDefinition
from ..factory import ProceduralDataset, register_dataset

DATASET_NAME = "linear_programming"


@dataclass
class LinearProgrammingConfig:
    min_coeff: int = 1
    max_coeff: int = 10
    num_constraints: int = 3
    seed: Optional[int] = None
    size: int = 500

    def validate(self) -> None:
        assert self.size > 0, "size must be positive"
        assert self.min_coeff >= 1, "min_coeff must be >= 1"
        assert self.max_coeff >= self.min_coeff, "max_coeff must be >= min_coeff"
        assert 2 <= self.num_constraints <= 6, "num_constraints must be between 2 and 6"


class LinearProgrammingDataset(ProceduralDataset):
    """2-variable LP problems with backward construction from a known optimal vertex."""

    def __init__(self, config: LinearProgrammingConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def __getitem__(self, idx: int) -> dict:
        rng = random.Random(self.seed + idx)

        x_opt = rng.randint(1, self.config.max_coeff)
        y_opt = rng.randint(1, self.config.max_coeff)
        c1 = rng.randint(self.config.min_coeff, self.config.max_coeff)
        c2 = rng.randint(self.config.min_coeff, self.config.max_coeff)
        opt_val = c1 * x_opt + c2 * y_opt

        constraints = []
        for _ in range(self.config.num_constraints):
            a = rng.randint(self.config.min_coeff, self.config.max_coeff)
            b = rng.randint(self.config.min_coeff, self.config.max_coeff)
            rhs = a * x_opt + b * y_opt + rng.randint(0, 5)
            constraints.append((a, b, rhs))

        tight_a = rng.randint(self.config.min_coeff, self.config.max_coeff)
        tight_b = rng.randint(self.config.min_coeff, self.config.max_coeff)
        tight_rhs = tight_a * x_opt + tight_b * y_opt
        constraints[0] = (tight_a, tight_b, tight_rhs)

        constraint_strs = []
        for a, b, rhs in constraints:
            constraint_strs.append(f"  {a}x + {b}y <= {rhs}")
        constraint_strs.append("  x >= 0")
        constraint_strs.append("  y >= 0")
        constraints_text = "\n".join(constraint_strs)

        question = (
            f"Maximize {c1}x + {c2}y subject to:\n{constraints_text}\n"
            f"What is the maximum value of the objective function? "
            f"Give your answer as a single integer."
        )

        return {
            "question": question,
            "answer": str(opt_val),
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "optimal_point": (x_opt, y_opt),
                "difficulty": {
                    "num_constraints": self.config.num_constraints,
                    "max_coeff": self.config.max_coeff,
                },
            },
        }

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        if answer is None:
            return 0.0
        try:
            return 1.0 if int(answer.strip()) == int(entry["answer"]) else 0.0
        except (ValueError, TypeError):
            return 0.0


class LinearProgrammingCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(LinearProgrammingCurriculum.__name__, LinearProgrammingConfig)
        self._define_attributes(
            ScalarAttributeDefinition(
                name="num_constraints",
                field_name="num_constraints",
                levels=[2, 3, 4, 5],
                description="Number of inequality constraints",
            ),
            ScalarAttributeDefinition(
                name="max_coeff",
                field_name="max_coeff",
                levels=[5, 10, 20, 50],
                description="Maximum coefficient value",
            ),
        )


register_dataset(DATASET_NAME, LinearProgrammingDataset, LinearProgrammingConfig, LinearProgrammingCurriculum)
