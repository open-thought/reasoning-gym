import math
import random
from dataclasses import dataclass, field
from typing import Any, Optional

from ..coaching import BaseCurriculum, RangeAttributeDefinition
from ..factory import ProceduralDataset, register_dataset

DATASET_NAME = "number_theory"

TASK_TYPES = ("mod_arith", "mod_exp", "totient", "crt", "mod_inverse", "diophantine")


def euler_totient(n: int) -> int:
    result = n
    p = 2
    temp = n
    while p * p <= temp:
        if temp % p == 0:
            while temp % p == 0:
                temp //= p
            result -= result // p
        p += 1
    if temp > 1:
        result -= result // temp
    return result


def extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    if a == 0:
        return b, 0, 1
    g, x1, y1 = extended_gcd(b % a, a)
    return g, y1 - (b // a) * x1, x1


@dataclass
class NumberTheoryConfig:
    min_value: int = 2
    max_value: int = 50
    max_exp: int = 20
    task_types: tuple[str, ...] = TASK_TYPES
    task_weights: list[float] = field(default_factory=lambda: [0.2, 0.2, 0.15, 0.15, 0.15, 0.15])
    seed: Optional[int] = None
    size: int = 500

    def validate(self) -> None:
        assert self.size > 0, "size must be positive"
        assert self.min_value >= 2, "min_value must be >= 2"
        assert self.max_value >= self.min_value, "max_value must be >= min_value"
        assert self.max_exp >= 2, "max_exp must be >= 2"
        assert len(self.task_types) > 0, "must have at least one task type"
        assert all(t in TASK_TYPES for t in self.task_types), f"invalid task type"
        assert len(self.task_weights) == len(self.task_types), "weights must match types"


class NumberTheoryDataset(ProceduralDataset):
    def __init__(self, config: NumberTheoryConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _make_mod_arith(self, rng: random.Random) -> dict:
        a = rng.randint(self.config.min_value, self.config.max_value * 5)
        m = rng.randint(self.config.min_value, self.config.max_value)
        answer = a % m
        question = f"What is {a} mod {m}? Give your answer as a single integer."
        return {"question": question, "answer": str(answer), "task_type": "mod_arith"}

    def _make_mod_exp(self, rng: random.Random) -> dict:
        base = rng.randint(2, self.config.max_value)
        exp = rng.randint(2, self.config.max_exp)
        mod = rng.randint(self.config.min_value, self.config.max_value)
        answer = pow(base, exp, mod)
        question = f"What is {base}^{exp} mod {mod}? Give your answer as a single integer."
        return {"question": question, "answer": str(answer), "task_type": "mod_exp"}

    def _make_totient(self, rng: random.Random) -> dict:
        n = rng.randint(self.config.min_value, self.config.max_value)
        answer = euler_totient(n)
        question = (
            f"Compute Euler's totient function φ({n}), i.e., the count of integers "
            f"from 1 to {n} that are coprime to {n}. Give your answer as a single integer."
        )
        return {"question": question, "answer": str(answer), "task_type": "totient"}

    def _make_crt(self, rng: random.Random) -> dict:
        m1 = rng.randint(2, 10)
        m2 = rng.randint(2, 10)
        while math.gcd(m1, m2) != 1:
            m2 = rng.randint(2, 10)
        r1 = rng.randint(0, m1 - 1)
        r2 = rng.randint(0, m2 - 1)

        for x in range(m1 * m2):
            if x % m1 == r1 and x % m2 == r2:
                answer = x
                break

        question = (
            f"Find the smallest non-negative integer x such that:\n"
            f"  x ≡ {r1} (mod {m1})\n"
            f"  x ≡ {r2} (mod {m2})\n"
            f"Give your answer as a single integer."
        )
        return {"question": question, "answer": str(answer), "task_type": "crt"}

    def _make_mod_inverse(self, rng: random.Random) -> dict:
        m = rng.randint(3, self.config.max_value)
        a = rng.randint(2, m - 1)
        while math.gcd(a, m) != 1:
            a = rng.randint(2, m - 1)
        answer = pow(a, -1, m)
        question = (
            f"Find the modular inverse of {a} modulo {m}, i.e., find x such that "
            f"{a} * x ≡ 1 (mod {m}). Give x as a single integer (0 ≤ x < {m})."
        )
        return {"question": question, "answer": str(answer), "task_type": "mod_inverse"}

    def _make_diophantine(self, rng: random.Random) -> dict:
        a = rng.randint(2, self.config.max_value)
        b = rng.randint(2, self.config.max_value)
        g = math.gcd(a, b)
        c = g * rng.randint(1, 5)

        _, x0, y0 = extended_gcd(a, b)
        x0 *= c // g
        y0 *= c // g
        answer = f"x={x0}, y={y0}"
        question = (
            f"Find one integer solution (x, y) to the equation {a}x + {b}y = {c}. "
            f"Format your answer as: x=<value>, y=<value>"
        )
        return {"question": question, "answer": answer, "task_type": "diophantine", "a": a, "b": b, "c": c}

    def __getitem__(self, idx: int) -> dict:
        rng = random.Random(self.seed + idx)
        task_type = rng.choices(self.config.task_types, weights=self.config.task_weights, k=1)[0]

        generators = {
            "mod_arith": self._make_mod_arith,
            "mod_exp": self._make_mod_exp,
            "totient": self._make_totient,
            "crt": self._make_crt,
            "mod_inverse": self._make_mod_inverse,
            "diophantine": self._make_diophantine,
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
                    "min_value": self.config.min_value,
                    "max_value": self.config.max_value,
                },
                **({"a": result["a"], "b": result["b"], "c": result["c"]} if "a" in result else {}),
            },
        }

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        if answer is None:
            return 0.0
        oracle = entry["answer"]
        if answer.strip() == oracle.strip():
            return 1.0
        task_type = entry["metadata"]["task_type"]
        if task_type == "diophantine":
            try:
                parts = {}
                for part in answer.strip().split(","):
                    k, v = part.split("=")
                    parts[k.strip()] = int(v.strip())
                x, y = parts["x"], parts["y"]
                a = entry["metadata"]["a"]
                b = entry["metadata"]["b"]
                c = entry["metadata"]["c"]
                if a * x + b * y == c:
                    return 1.0
                return 0.0
            except (ValueError, KeyError, TypeError):
                return 0.0
        try:
            return 1.0 if int(answer.strip()) == int(oracle.strip()) else 0.0
        except ValueError:
            return 0.0


class NumberTheoryCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(NumberTheoryCurriculum.__name__, NumberTheoryConfig)
        self._define_attributes(
            RangeAttributeDefinition(
                name="value_range",
                levels=[10, 50, 100, 500],
                lower_field_name="min_value",
                upper_field_name="max_value",
                description="Range for numbers in problems",
            ),
        )


register_dataset(DATASET_NAME, NumberTheoryDataset, NumberTheoryConfig, NumberTheoryCurriculum)
