import json
import random
from dataclasses import dataclass, field
from typing import Any, Optional

from ..coaching import BaseCurriculum, ScalarAttributeDefinition
from ..factory import ProceduralDataset, register_dataset

DATASET_NAME = "linear_algebra"

TASK_TYPES = ("matrix_multiply", "determinant", "inverse", "solve_system", "eigenvalues")


@dataclass
class LinearAlgebraConfig:
    min_dim: int = 2
    max_dim: int = 3
    min_value: int = -5
    max_value: int = 5
    task_types: tuple[str, ...] = TASK_TYPES
    task_weights: list[float] = field(default_factory=lambda: [0.25, 0.2, 0.2, 0.2, 0.15])
    seed: Optional[int] = None
    size: int = 500

    def validate(self) -> None:
        assert self.size > 0, "size must be positive"
        assert self.min_dim >= 2, "min_dim must be >= 2"
        assert self.max_dim >= self.min_dim, "max_dim must be >= min_dim"
        assert self.max_dim <= 4, "max_dim must be <= 4"
        assert len(self.task_types) > 0, "must have at least one task type"
        assert all(t in TASK_TYPES for t in self.task_types), f"invalid task type"
        assert len(self.task_weights) == len(self.task_types), "weights must match types"


def _mat_str(m: list[list[int]]) -> str:
    rows = ["[" + ", ".join(str(x) for x in row) + "]" for row in m]
    return "[" + ", ".join(rows) + "]"


def _mat_mult(a: list[list[int]], b: list[list[int]]) -> list[list[int]]:
    n, m, p = len(a), len(a[0]), len(b[0])
    result = [[0] * p for _ in range(n)]
    for i in range(n):
        for j in range(p):
            for k in range(m):
                result[i][j] += a[i][k] * b[k][j]
    return result


def _det(m: list[list[int]]) -> int:
    n = len(m)
    if n == 1:
        return m[0][0]
    if n == 2:
        return m[0][0] * m[1][1] - m[0][1] * m[1][0]
    result = 0
    for j in range(n):
        sub = [[m[i][k] for k in range(n) if k != j] for i in range(1, n)]
        result += ((-1) ** j) * m[0][j] * _det(sub)
    return result


def _adjugate_2x2(m: list[list[int]]) -> list[list[int]]:
    return [[m[1][1], -m[0][1]], [-m[1][0], m[0][0]]]


class LinearAlgebraDataset(ProceduralDataset):
    def __init__(self, config: LinearAlgebraConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _gen_matrix(self, rng: random.Random, rows: int, cols: int) -> list[list[int]]:
        return [
            [rng.randint(self.config.min_value, self.config.max_value) for _ in range(cols)]
            for _ in range(rows)
        ]

    def _make_matrix_multiply(self, rng: random.Random) -> dict:
        n = rng.randint(self.config.min_dim, self.config.max_dim)
        m = rng.randint(self.config.min_dim, self.config.max_dim)
        p = rng.randint(self.config.min_dim, self.config.max_dim)
        a = self._gen_matrix(rng, n, m)
        b = self._gen_matrix(rng, m, p)
        result = _mat_mult(a, b)
        question = (
            f"Multiply the matrices A = {_mat_str(a)} and B = {_mat_str(b)}. "
            f"Give the result as a nested list, e.g. [[1, 2], [3, 4]]."
        )
        return {"question": question, "answer": _mat_str(result), "task_type": "matrix_multiply"}

    def _make_determinant(self, rng: random.Random) -> dict:
        n = rng.randint(self.config.min_dim, min(self.config.max_dim, 3))
        m = self._gen_matrix(rng, n, n)
        result = _det(m)
        question = (
            f"Find the determinant of the matrix {_mat_str(m)}. "
            f"Give your answer as a single integer."
        )
        return {"question": question, "answer": str(result), "task_type": "determinant"}

    def _make_inverse(self, rng: random.Random) -> dict:
        for _ in range(100):
            m = self._gen_matrix(rng, 2, 2)
            d = _det(m)
            if d != 0 and all(x % d == 0 for row in _adjugate_2x2(m) for x in row):
                adj = _adjugate_2x2(m)
                inv = [[x // d for x in row] for row in adj]
                question = (
                    f"Find the inverse of the 2x2 matrix {_mat_str(m)}. "
                    f"Give the result as a nested list of integers, e.g. [[1, 2], [3, 4]]."
                )
                return {"question": question, "answer": _mat_str(inv), "task_type": "inverse"}
        m = [[1, 0], [0, 1]]
        question = f"Find the inverse of the 2x2 matrix {_mat_str(m)}. Give the result as a nested list."
        return {"question": question, "answer": _mat_str(m), "task_type": "inverse"}

    def _make_solve_system(self, rng: random.Random) -> dict:
        n = 2
        x_sol = [rng.randint(-5, 5) for _ in range(n)]
        for _ in range(100):
            a = self._gen_matrix(rng, n, n)
            d = _det(a)
            if d != 0:
                break
        else:
            a = [[1, 0], [0, 1]]

        b = [sum(a[i][j] * x_sol[j] for j in range(n)) for i in range(n)]
        vars_ = ["x", "y"]
        eqs = []
        for i in range(n):
            parts = []
            for j in range(n):
                coef = a[i][j]
                if coef == 0:
                    continue
                if coef == 1:
                    parts.append(f"{vars_[j]}")
                elif coef == -1:
                    parts.append(f"-{vars_[j]}")
                else:
                    parts.append(f"{coef}{vars_[j]}")
            eq = " + ".join(parts).replace("+ -", "- ")
            eqs.append(f"  {eq} = {b[i]}")
        eq_str = "\n".join(eqs)
        answer = ", ".join(f"{vars_[i]}={x_sol[i]}" for i in range(n))
        question = (
            f"Solve the following system of linear equations:\n{eq_str}\n"
            f"Give your answer in the format: x=<value>, y=<value>"
        )
        return {
            "question": question,
            "answer": answer,
            "task_type": "solve_system",
            "matrix": a,
            "b": b,
        }

    def _make_eigenvalues(self, rng: random.Random) -> dict:
        e1 = rng.randint(-5, 5)
        e2 = rng.randint(-5, 5)
        m = [[e1, 0], [0, e2]]
        p_det = 1
        for _ in range(3):
            shear = self._gen_matrix(rng, 2, 2)
            d = _det(shear)
            if abs(d) == 1:
                inv_d = d
                adj = _adjugate_2x2(shear)
                shear_inv = [[x * inv_d for x in row] for row in adj]
                temp = _mat_mult(shear, m)
                m = _mat_mult(temp, shear_inv)
                break

        eigenvals = sorted([e1, e2])
        answer = ", ".join(str(e) for e in eigenvals)
        question = (
            f"Find the eigenvalues of the 2x2 matrix {_mat_str(m)}. "
            f"List them separated by commas in ascending order."
        )
        return {"question": question, "answer": answer, "task_type": "eigenvalues"}

    def __getitem__(self, idx: int) -> dict:
        rng = random.Random(self.seed + idx)
        task_type = rng.choices(self.config.task_types, weights=self.config.task_weights, k=1)[0]

        generators = {
            "matrix_multiply": self._make_matrix_multiply,
            "determinant": self._make_determinant,
            "inverse": self._make_inverse,
            "solve_system": self._make_solve_system,
            "eigenvalues": self._make_eigenvalues,
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
                    "min_dim": self.config.min_dim,
                    "max_dim": self.config.max_dim,
                },
                **({"matrix": result["matrix"], "b": result["b"]} if "matrix" in result else {}),
            },
        }

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        if answer is None:
            return 0.0
        oracle = entry["answer"]
        if answer.strip() == oracle.strip():
            return 1.0
        task_type = entry["metadata"]["task_type"]

        if task_type == "solve_system":
            try:
                parts = {}
                for part in answer.strip().split(","):
                    k, v = part.split("=")
                    parts[k.strip()] = int(v.strip())
                x, y = parts["x"], parts["y"]
                mat = entry["metadata"]["matrix"]
                b = entry["metadata"]["b"]
                if mat[0][0] * x + mat[0][1] * y == b[0] and mat[1][0] * x + mat[1][1] * y == b[1]:
                    return 1.0
                return 0.0
            except (ValueError, KeyError, TypeError):
                return 0.0

        if task_type in ("determinant",):
            try:
                return 1.0 if int(answer.strip()) == int(oracle.strip()) else 0.0
            except ValueError:
                return 0.0

        if task_type == "eigenvalues":
            try:
                a_vals = sorted(int(x.strip()) for x in answer.split(","))
                o_vals = sorted(int(x.strip()) for x in oracle.split(","))
                return 1.0 if a_vals == o_vals else 0.0
            except (ValueError, TypeError):
                return 0.0

        try:
            a_mat = json.loads(answer.strip())
            o_mat = json.loads(oracle.strip())
            return 1.0 if a_mat == o_mat else 0.0
        except (json.JSONDecodeError, ValueError):
            return 0.0


class LinearAlgebraCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(LinearAlgebraCurriculum.__name__, LinearAlgebraConfig)
        self._define_attributes(
            ScalarAttributeDefinition(
                name="max_dim",
                field_name="max_dim",
                levels=[2, 3, 4],
                description="Maximum matrix dimension",
            ),
        )


register_dataset(DATASET_NAME, LinearAlgebraDataset, LinearAlgebraConfig, LinearAlgebraCurriculum)
