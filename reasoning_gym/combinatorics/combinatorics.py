import math
import random
from dataclasses import dataclass, field
from typing import Any, Optional

from ..coaching import BaseCurriculum, RangeAttributeDefinition
from ..factory import ProceduralDataset, register_dataset

DATASET_NAME = "combinatorics"

TASK_TYPES = ("ncr", "npr", "permutations_repetition", "inclusion_exclusion", "stars_and_bars", "pigeonhole")


@dataclass
class CombinatoricsConfig:
    min_n: int = 5
    max_n: int = 15
    task_types: tuple[str, ...] = TASK_TYPES
    task_weights: list[float] = field(
        default_factory=lambda: [0.2, 0.15, 0.2, 0.2, 0.15, 0.1]
    )
    seed: Optional[int] = None
    size: int = 500

    def validate(self) -> None:
        assert self.size > 0, "size must be positive"
        assert self.min_n >= 2, "min_n must be >= 2"
        assert self.max_n >= self.min_n, "max_n must be >= min_n"
        assert len(self.task_types) > 0, "must have at least one task type"
        assert all(t in TASK_TYPES for t in self.task_types), f"invalid task type"
        assert len(self.task_weights) == len(self.task_types), "weights must match types"


class CombinatoricsDataset(ProceduralDataset):
    def __init__(self, config: CombinatoricsConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _make_ncr(self, rng: random.Random) -> dict:
        n = rng.randint(self.config.min_n, self.config.max_n)
        k = rng.randint(1, n - 1)
        answer = math.comb(n, k)
        question = (
            f"How many ways can you choose {k} items from a set of {n} items? "
            f"Give your answer as a single integer."
        )
        return {"question": question, "answer": str(answer), "task_type": "ncr"}

    def _make_npr(self, rng: random.Random) -> dict:
        n = rng.randint(self.config.min_n, self.config.max_n)
        k = rng.randint(1, min(n - 1, 6))
        answer = math.perm(n, k)
        question = (
            f"How many ways can you arrange {k} items chosen from {n} distinct items "
            f"(order matters)? Give your answer as a single integer."
        )
        return {"question": question, "answer": str(answer), "task_type": "npr"}

    def _make_permutations_repetition(self, rng: random.Random) -> dict:
        letters = []
        num_distinct = rng.randint(2, 4)
        pool = "ABCDEFGH"
        chosen = rng.sample(pool, num_distinct)
        counts = {}
        for ch in chosen:
            c = rng.randint(1, 4)
            counts[ch] = c
            letters.extend([ch] * c)
        rng.shuffle(letters)
        word = "".join(letters)

        numerator = math.factorial(len(word))
        denominator = 1
        for c in counts.values():
            denominator *= math.factorial(c)
        answer = numerator // denominator

        count_desc = ", ".join(f"'{k}' appears {v} time(s)" for k, v in sorted(counts.items()))
        question = (
            f"How many distinct arrangements can be made from the letters of '{word}'? "
            f"({count_desc}) Give your answer as a single integer."
        )
        return {"question": question, "answer": str(answer), "task_type": "permutations_repetition"}

    def _make_inclusion_exclusion(self, rng: random.Random) -> dict:
        total = rng.randint(50, 200)
        a_count = rng.randint(total // 4, total * 3 // 4)
        b_count = rng.randint(total // 4, total * 3 // 4)
        max_both = min(a_count, b_count, total)
        min_both = max(0, a_count + b_count - total)
        both = rng.randint(min_both, max_both)
        neither = total - (a_count + b_count - both)

        activity_a = rng.choice(["play soccer", "like tea", "study math", "read fiction"])
        activity_b = rng.choice(["play chess", "like coffee", "study science", "read poetry"])

        question = (
            f"In a group of {total} people, {a_count} {activity_a}, {b_count} {activity_b}, "
            f"and {both} do both. How many people do neither? "
            f"Give your answer as a single integer."
        )
        return {"question": question, "answer": str(neither), "task_type": "inclusion_exclusion"}

    def _make_stars_and_bars(self, rng: random.Random) -> dict:
        n = rng.randint(self.config.min_n, self.config.max_n)
        k = rng.randint(2, 5)
        answer = math.comb(n + k - 1, k - 1)
        question = (
            f"How many ways can you distribute {n} identical balls into {k} distinct boxes "
            f"(each box can hold any number of balls)? Give your answer as a single integer."
        )
        return {"question": question, "answer": str(answer), "task_type": "stars_and_bars"}

    def _make_pigeonhole(self, rng: random.Random) -> dict:
        boxes = rng.randint(3, 20)
        extra = rng.randint(1, 10)
        items = boxes * extra + rng.randint(1, boxes - 1)
        answer = (items + boxes - 1) // boxes  # ceiling division

        question = (
            f"If {items} items are placed into {boxes} boxes, what is the minimum number of items "
            f"that must be in at least one box? Give your answer as a single integer."
        )
        return {"question": question, "answer": str(answer), "task_type": "pigeonhole"}

    def __getitem__(self, idx: int) -> dict:
        rng = random.Random(self.seed + idx)
        task_type = rng.choices(self.config.task_types, weights=self.config.task_weights, k=1)[0]

        generators = {
            "ncr": self._make_ncr,
            "npr": self._make_npr,
            "permutations_repetition": self._make_permutations_repetition,
            "inclusion_exclusion": self._make_inclusion_exclusion,
            "stars_and_bars": self._make_stars_and_bars,
            "pigeonhole": self._make_pigeonhole,
        }
        result = generators[task_type](rng)
        return {
            "question": result["question"],
            "answer": result["answer"],
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "task_type": result["task_type"],
                "difficulty": {"min_n": self.config.min_n, "max_n": self.config.max_n},
            },
        }


class CombinatoricsCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(CombinatoricsCurriculum.__name__, CombinatoricsConfig)
        self._define_attributes(
            RangeAttributeDefinition(
                name="n_range",
                levels=[5, 10, 20, 30],
                lower_field_name="min_n",
                upper_field_name="max_n",
                description="Range for n in combinatorial problems",
            ),
        )


register_dataset(DATASET_NAME, CombinatoricsDataset, CombinatoricsConfig, CombinatoricsCurriculum)
