import random
from dataclasses import dataclass
from typing import Any, Optional

from ..coaching import BaseCurriculum, RangeAttributeDefinition, ScalarAttributeDefinition
from ..factory import ProceduralDataset, register_dataset

DATASET_NAME = "knapsack"


@dataclass
class KnapsackConfig:
    min_items: int = 3
    max_items: int = 6
    min_weight: int = 1
    max_weight: int = 15
    min_value: int = 5
    max_value: int = 50
    min_capacity: int = 10
    max_capacity: int = 30
    seed: Optional[int] = None
    size: int = 500

    def validate(self) -> None:
        assert self.size > 0, "size must be positive"
        assert self.min_items >= 2, "min_items must be >= 2"
        assert self.max_items >= self.min_items, "max_items must be >= min_items"
        assert self.min_weight >= 1, "min_weight must be >= 1"
        assert self.max_weight >= self.min_weight, "max_weight must be >= min_weight"
        assert self.min_value >= 1, "min_value must be >= 1"
        assert self.max_value >= self.min_value, "max_value must be >= min_value"
        assert self.min_capacity >= 1, "min_capacity must be >= 1"
        assert self.max_capacity >= self.min_capacity, "max_capacity must be >= min_capacity"


def _solve_knapsack(weights: list[int], values: list[int], capacity: int) -> int:
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            dp[i][w] = dp[i - 1][w]
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
    return dp[n][capacity]


class KnapsackDataset(ProceduralDataset):
    def __init__(self, config: KnapsackConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def __getitem__(self, idx: int) -> dict:
        rng = random.Random(self.seed + idx)

        n = rng.randint(self.config.min_items, self.config.max_items)
        weights = [rng.randint(self.config.min_weight, self.config.max_weight) for _ in range(n)]
        values = [rng.randint(self.config.min_value, self.config.max_value) for _ in range(n)]
        capacity = rng.randint(self.config.min_capacity, self.config.max_capacity)

        opt_val = _solve_knapsack(weights, values, capacity)

        items_str = ", ".join(f"(weight={w}, value={v})" for w, v in zip(weights, values))
        question = (
            f"You have a knapsack with capacity {capacity}. "
            f"You have the following items: {items_str}. "
            f"Each item can be used at most once. "
            f"What is the maximum total value you can carry? "
            f"Give your answer as a single integer."
        )

        return {
            "question": question,
            "answer": str(opt_val),
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "weights": weights,
                "values": values,
                "capacity": capacity,
                "difficulty": {
                    "min_items": self.config.min_items,
                    "max_items": self.config.max_items,
                },
            },
        }


class KnapsackCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(KnapsackCurriculum.__name__, KnapsackConfig)
        self._define_attributes(
            RangeAttributeDefinition(
                name="item_count",
                levels=[3, 6, 10, 15],
                lower_field_name="min_items",
                upper_field_name="max_items",
                description="Number of items",
            ),
            RangeAttributeDefinition(
                name="capacity",
                levels=[10, 30, 50, 100],
                lower_field_name="min_capacity",
                upper_field_name="max_capacity",
                description="Knapsack capacity range",
            ),
        )


register_dataset(DATASET_NAME, KnapsackDataset, KnapsackConfig, KnapsackCurriculum)
