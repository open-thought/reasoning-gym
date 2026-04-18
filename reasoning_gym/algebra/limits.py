import random
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Optional

from ..coaching import BaseCurriculum, ScalarAttributeDefinition
from ..factory import ProceduralDataset, register_dataset

DATASET_NAME = "limits"

TASK_TYPES = ("polynomial_cancel", "rational_infinity", "direct_sub", "squeeze")


@dataclass
class LimitsConfig:
    max_coeff: int = 10
    max_degree: int = 3
    task_types: tuple[str, ...] = TASK_TYPES
    task_weights: list[float] = field(default_factory=lambda: [0.3, 0.3, 0.2, 0.2])
    seed: Optional[int] = None
    size: int = 500

    def validate(self) -> None:
        assert self.size > 0, "size must be positive"
        assert self.max_coeff >= 1, "max_coeff must be >= 1"
        assert self.max_degree >= 1, "max_degree must be >= 1"
        assert len(self.task_types) > 0, "must have at least one task type"
        assert all(t in TASK_TYPES for t in self.task_types), f"invalid task type"
        assert len(self.task_weights) == len(self.task_types), "weights must match types"


def _poly_str(coeffs: list[int], var: str = "x") -> str:
    """coeffs[i] is coefficient of x^i"""
    parts = []
    for i in range(len(coeffs) - 1, -1, -1):
        c = coeffs[i]
        if c == 0:
            continue
        if i == 0:
            parts.append(str(c))
        elif i == 1:
            if c == 1:
                parts.append(var)
            elif c == -1:
                parts.append(f"-{var}")
            else:
                parts.append(f"{c}*{var}")
        else:
            if c == 1:
                parts.append(f"{var}^{i}")
            elif c == -1:
                parts.append(f"-{var}^{i}")
            else:
                parts.append(f"{c}*{var}^{i}")
    if not parts:
        return "0"
    result = parts[0]
    for p in parts[1:]:
        if p.startswith("-"):
            result += " - " + p[1:]
        else:
            result += " + " + p
    return result


class LimitsDataset(ProceduralDataset):
    def __init__(self, config: LimitsConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _make_polynomial_cancel(self, rng: random.Random) -> dict:
        a = rng.randint(-self.config.max_coeff, self.config.max_coeff)
        if a == 0:
            a = 1
        b = rng.randint(1, self.config.max_coeff)
        c = rng.randint(1, self.config.max_coeff) * rng.choice([1, -1])

        num_val = b * a + c
        answer_frac = Fraction(num_val, 1)
        answer = str(answer_frac)

        num_expr = f"{b}*x" if b != 1 else "x"
        if c > 0:
            num_expr += f" + {c}"
        elif c < 0:
            num_expr += f" - {abs(c)}"

        denom_expr = f"x - {a}" if a >= 0 else f"x + {abs(a)}"
        full_num = f"({num_expr}) * ({denom_expr})"

        question = (
            f"Find the limit as x approaches {a} of {full_num} / ({denom_expr}). "
            f"Give your answer as an integer or simplified fraction."
        )
        return {"question": question, "answer": answer, "task_type": "polynomial_cancel"}

    def _make_rational_infinity(self, rng: random.Random) -> dict:
        deg = rng.randint(1, self.config.max_degree)
        num_lead = rng.randint(1, self.config.max_coeff) * rng.choice([1, -1])
        den_lead = rng.randint(1, self.config.max_coeff) * rng.choice([1, -1])

        num_coeffs = [rng.randint(-self.config.max_coeff, self.config.max_coeff) for _ in range(deg)]
        num_coeffs.append(num_lead)
        den_coeffs = [rng.randint(-self.config.max_coeff, self.config.max_coeff) for _ in range(deg)]
        den_coeffs.append(den_lead)

        answer_frac = Fraction(num_lead, den_lead)
        answer = str(answer_frac)

        num_str = _poly_str(num_coeffs)
        den_str = _poly_str(den_coeffs)
        question = (
            f"Find the limit as x approaches infinity of ({num_str}) / ({den_str}). "
            f"Give your answer as an integer or simplified fraction."
        )
        return {"question": question, "answer": answer, "task_type": "rational_infinity"}

    def _make_direct_sub(self, rng: random.Random) -> dict:
        a = rng.randint(1, 5)
        deg = rng.randint(1, self.config.max_degree)
        coeffs = [rng.randint(-self.config.max_coeff, self.config.max_coeff) for _ in range(deg + 1)]
        if coeffs[-1] == 0:
            coeffs[-1] = 1

        val = sum(coeffs[i] * (a ** i) for i in range(len(coeffs)))
        answer = str(val)
        poly = _poly_str(coeffs)
        question = (
            f"Find the limit as x approaches {a} of ({poly}). "
            f"Give your answer as a single integer."
        )
        return {"question": question, "answer": answer, "task_type": "direct_sub"}

    def _make_squeeze(self, rng: random.Random) -> dict:
        L = rng.randint(-5, 5)
        a = rng.randint(-3, 3)
        k = rng.randint(1, 3)

        question = (
            f"Suppose that for all x near {a}, we have:\n"
            f"  {L} - (x - {a})^{2 * k} ≤ f(x) ≤ {L} + (x - {a})^{2 * k}\n"
            f"Find the limit of f(x) as x approaches {a}. Give your answer as a single integer."
        )
        return {"question": question, "answer": str(L), "task_type": "squeeze"}

    def __getitem__(self, idx: int) -> dict:
        rng = random.Random(self.seed + idx)
        task_type = rng.choices(self.config.task_types, weights=self.config.task_weights, k=1)[0]

        generators = {
            "polynomial_cancel": self._make_polynomial_cancel,
            "rational_infinity": self._make_rational_infinity,
            "direct_sub": self._make_direct_sub,
            "squeeze": self._make_squeeze,
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
                    "max_coeff": self.config.max_coeff,
                    "max_degree": self.config.max_degree,
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
            a_frac = Fraction(answer.strip())
            o_frac = Fraction(oracle.strip())
            return 1.0 if a_frac == o_frac else 0.0
        except (ValueError, ZeroDivisionError):
            try:
                return 1.0 if float(answer.strip()) == float(oracle.strip()) else 0.0
            except ValueError:
                return 0.0


class LimitsCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(LimitsCurriculum.__name__, LimitsConfig)
        self._define_attributes(
            ScalarAttributeDefinition(
                name="max_coeff",
                field_name="max_coeff",
                levels=[5, 10, 20, 50],
                description="Maximum coefficient magnitude",
            ),
            ScalarAttributeDefinition(
                name="max_degree",
                field_name="max_degree",
                levels=[1, 2, 3, 4],
                description="Maximum polynomial degree",
            ),
        )


register_dataset(DATASET_NAME, LimitsDataset, LimitsConfig, LimitsCurriculum)
