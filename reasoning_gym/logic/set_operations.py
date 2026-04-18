import random
from dataclasses import dataclass, field
from typing import Any, Optional

from ..coaching import BaseCurriculum, RangeAttributeDefinition
from ..factory import ProceduralDataset, register_dataset

DATASET_NAME = "set_operations"

TASK_TYPES = (
    "union",
    "intersection",
    "difference",
    "symmetric_difference",
    "cardinality",
    "power_set_size",
    "complement",
    "chained",
)


@dataclass
class SetOperationsConfig:
    min_set_size: int = 3
    max_set_size: int = 8
    min_value: int = 1
    max_value: int = 20
    task_types: tuple[str, ...] = TASK_TYPES
    task_weights: list[float] = field(default_factory=lambda: [0.15, 0.15, 0.12, 0.12, 0.12, 0.1, 0.12, 0.12])
    seed: Optional[int] = None
    size: int = 500

    def validate(self) -> None:
        assert self.size > 0, "size must be positive"
        assert self.min_set_size >= 1, "min_set_size must be >= 1"
        assert self.max_set_size >= self.min_set_size, "max_set_size must be >= min_set_size"
        assert self.max_value > self.min_value, "max_value must be > min_value"
        assert len(self.task_types) > 0, "must have at least one task type"
        assert all(t in TASK_TYPES for t in self.task_types), f"invalid task type"
        assert len(self.task_weights) == len(self.task_types), "weights must match types"


def _fmt_set(s: set) -> str:
    return "{" + ", ".join(str(x) for x in sorted(s)) + "}"


class SetOperationsDataset(ProceduralDataset):
    def __init__(self, config: SetOperationsConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _gen_set(self, rng: random.Random) -> set:
        n = rng.randint(self.config.min_set_size, self.config.max_set_size)
        return set(rng.sample(range(self.config.min_value, self.config.max_value + 1), n))

    def _make_union(self, rng: random.Random) -> dict:
        a, b = self._gen_set(rng), self._gen_set(rng)
        result = a | b
        question = f"Given A = {_fmt_set(a)} and B = {_fmt_set(b)}, find A ∪ B (the union)."
        return {"question": question, "answer": _fmt_set(result), "task_type": "union"}

    def _make_intersection(self, rng: random.Random) -> dict:
        a, b = self._gen_set(rng), self._gen_set(rng)
        result = a & b
        question = f"Given A = {_fmt_set(a)} and B = {_fmt_set(b)}, find A ∩ B (the intersection)."
        return {"question": question, "answer": _fmt_set(result), "task_type": "intersection"}

    def _make_difference(self, rng: random.Random) -> dict:
        a, b = self._gen_set(rng), self._gen_set(rng)
        result = a - b
        question = f"Given A = {_fmt_set(a)} and B = {_fmt_set(b)}, find A \\ B (elements in A but not in B)."
        return {"question": question, "answer": _fmt_set(result), "task_type": "difference"}

    def _make_symmetric_difference(self, rng: random.Random) -> dict:
        a, b = self._gen_set(rng), self._gen_set(rng)
        result = a ^ b
        question = f"Given A = {_fmt_set(a)} and B = {_fmt_set(b)}, find A △ B (the symmetric difference)."
        return {"question": question, "answer": _fmt_set(result), "task_type": "symmetric_difference"}

    def _make_cardinality(self, rng: random.Random) -> dict:
        a_size = rng.randint(5, 30)
        b_size = rng.randint(5, 30)
        both = rng.randint(0, min(a_size, b_size))
        union_size = a_size + b_size - both
        question = (
            f"If |A| = {a_size}, |B| = {b_size}, and |A ∩ B| = {both}, what is |A ∪ B|? "
            f"Give your answer as a single integer."
        )
        return {"question": question, "answer": str(union_size), "task_type": "cardinality"}

    def _make_power_set_size(self, rng: random.Random) -> dict:
        n = rng.randint(2, 8)
        answer = 2**n
        question = f"How many subsets does a set with {n} elements have? Give your answer as a single integer."
        return {"question": question, "answer": str(answer), "task_type": "power_set_size"}

    def _make_complement(self, rng: random.Random) -> dict:
        u_max = rng.randint(8, 15)
        universe = set(range(1, u_max + 1))
        a = set(rng.sample(sorted(universe), rng.randint(2, u_max - 2)))
        result = universe - a
        question = (
            f"If the universal set U = {_fmt_set(universe)} and A = {_fmt_set(a)}, "
            f"find A' (the complement of A in U)."
        )
        return {"question": question, "answer": _fmt_set(result), "task_type": "complement"}

    def _make_chained(self, rng: random.Random) -> dict:
        a, b, c = self._gen_set(rng), self._gen_set(rng), self._gen_set(rng)
        op1 = rng.choice(["union", "intersection"])
        op2 = rng.choice(["union", "intersection"])
        op1_sym = "∪" if op1 == "union" else "∩"
        op2_sym = "∪" if op2 == "union" else "∩"

        if op1 == "union":
            intermediate = a | b
        else:
            intermediate = a & b
        if op2 == "union":
            result = intermediate | c
        else:
            result = intermediate & c

        question = (
            f"Given A = {_fmt_set(a)}, B = {_fmt_set(b)}, C = {_fmt_set(c)}, " f"find (A {op1_sym} B) {op2_sym} C."
        )
        return {"question": question, "answer": _fmt_set(result), "task_type": "chained"}

    def __getitem__(self, idx: int) -> dict:
        rng = random.Random(self.seed + idx)
        task_type = rng.choices(self.config.task_types, weights=self.config.task_weights, k=1)[0]

        generators = {
            "union": self._make_union,
            "intersection": self._make_intersection,
            "difference": self._make_difference,
            "symmetric_difference": self._make_symmetric_difference,
            "cardinality": self._make_cardinality,
            "power_set_size": self._make_power_set_size,
            "complement": self._make_complement,
            "chained": self._make_chained,
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
                    "min_set_size": self.config.min_set_size,
                    "max_set_size": self.config.max_set_size,
                },
            },
        }

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        if answer is None:
            return 0.0
        oracle = entry["answer"]
        if answer.strip() == oracle.strip():
            return 1.0
        task_type = entry["metadata"]["task_type"]
        if task_type in ("cardinality", "power_set_size"):
            try:
                return 1.0 if int(answer.strip()) == int(oracle.strip()) else 0.0
            except ValueError:
                return 0.0
        try:
            parsed = set()
            inner = answer.strip().strip("{}")
            if inner:
                for x in inner.split(","):
                    parsed.add(int(x.strip()))
            oracle_set = set()
            oracle_inner = oracle.strip().strip("{}")
            if oracle_inner:
                for x in oracle_inner.split(","):
                    oracle_set.add(int(x.strip()))
            return 1.0 if parsed == oracle_set else 0.0
        except (ValueError, TypeError):
            return 0.0


class SetOperationsCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(SetOperationsCurriculum.__name__, SetOperationsConfig)
        self._define_attributes(
            RangeAttributeDefinition(
                name="set_size",
                levels=[3, 6, 10, 15],
                lower_field_name="min_set_size",
                upper_field_name="max_set_size",
                description="Size of generated sets",
            ),
        )


register_dataset(DATASET_NAME, SetOperationsDataset, SetOperationsConfig, SetOperationsCurriculum)
