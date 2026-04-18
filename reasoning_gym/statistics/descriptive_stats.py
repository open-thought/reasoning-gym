import math
import random
import statistics as stats_module
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

from ..coaching import BaseCurriculum, RangeAttributeDefinition, ScalarAttributeDefinition
from ..factory import ProceduralDataset, register_dataset

DATASET_NAME = "descriptive_stats"

TASK_TYPES = ("mean", "median", "mode", "weighted_mean", "std_dev", "percentile", "z_score")


@dataclass
class DescriptiveStatsConfig:
    min_data_size: int = 5
    max_data_size: int = 10
    min_value: int = 1
    max_value: int = 100
    decimal_places: int = 2
    task_types: tuple[str, ...] = TASK_TYPES
    task_weights: list[float] = field(default_factory=lambda: [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.1])
    seed: Optional[int] = None
    size: int = 500

    def validate(self) -> None:
        assert self.size > 0, "size must be positive"
        assert self.min_data_size >= 3, "min_data_size must be >= 3"
        assert self.max_data_size >= self.min_data_size, "max_data_size must be >= min_data_size"
        assert self.min_value < self.max_value, "min_value must be < max_value"
        assert len(self.task_types) > 0, "must have at least one task type"
        assert all(t in TASK_TYPES for t in self.task_types), f"invalid task type"
        assert len(self.task_weights) == len(self.task_types), "weights must match types"


class DescriptiveStatsDataset(ProceduralDataset):
    def __init__(self, config: DescriptiveStatsConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _gen_data(self, rng: random.Random) -> list[int]:
        n = rng.randint(self.config.min_data_size, self.config.max_data_size)
        return [rng.randint(self.config.min_value, self.config.max_value) for _ in range(n)]

    def _fmt(self, val: float) -> str:
        return f"{val:.{self.config.decimal_places}f}"

    def _make_mean(self, rng: random.Random) -> dict:
        data = self._gen_data(rng)
        answer = self._fmt(stats_module.mean(data))
        question = (
            f"Find the mean (average) of the following numbers: {data}. "
            f"Round your answer to {self.config.decimal_places} decimal places."
        )
        return {"question": question, "answer": answer, "task_type": "mean"}

    def _make_median(self, rng: random.Random) -> dict:
        data = self._gen_data(rng)
        answer = self._fmt(stats_module.median(data))
        question = (
            f"Find the median of the following numbers: {data}. "
            f"Round your answer to {self.config.decimal_places} decimal places."
        )
        return {"question": question, "answer": answer, "task_type": "median"}

    def _make_mode(self, rng: random.Random) -> dict:
        data = self._gen_data(rng)
        val = rng.choice(data)
        data.append(val)
        rng.shuffle(data)
        counts = Counter(data)
        max_count = max(counts.values())
        modes = sorted([k for k, v in counts.items() if v == max_count])
        answer = ", ".join(str(m) for m in modes)
        question = (
            f"Find the mode(s) of the following numbers: {data}. "
            f"If there are multiple modes, list them separated by commas in ascending order."
        )
        return {"question": question, "answer": answer, "task_type": "mode"}

    def _make_weighted_mean(self, rng: random.Random) -> dict:
        n = rng.randint(3, 5)
        values = [rng.randint(self.config.min_value, self.config.max_value) for _ in range(n)]
        raw_weights = [rng.randint(1, 10) for _ in range(n)]
        total_w = sum(raw_weights)
        weights = [w / total_w for w in raw_weights]

        result = sum(v * w for v, w in zip(values, weights))
        answer = self._fmt(result)

        pairs = ", ".join(f"value={v} weight={w:.2f}" for v, w in zip(values, weights))
        question = (
            f"Calculate the weighted mean of the following: {pairs}. "
            f"Round your answer to {self.config.decimal_places} decimal places."
        )
        return {"question": question, "answer": answer, "task_type": "weighted_mean"}

    def _make_std_dev(self, rng: random.Random) -> dict:
        data = self._gen_data(rng)
        answer = self._fmt(stats_module.pstdev(data))
        question = (
            f"Find the population standard deviation of the following numbers: {data}. "
            f"Round your answer to {self.config.decimal_places} decimal places."
        )
        return {"question": question, "answer": answer, "task_type": "std_dev"}

    def _make_percentile(self, rng: random.Random) -> dict:
        data = sorted(self._gen_data(rng))
        p = rng.choice([25, 50, 75, 90])
        n = len(data)
        rank = (p / 100) * (n - 1)
        lower = int(rank)
        frac = rank - lower
        if lower + 1 < n:
            val = data[lower] + frac * (data[lower + 1] - data[lower])
        else:
            val = data[lower]
        answer = self._fmt(val)
        question = (
            f"Find the {p}th percentile of the following numbers: {data}. "
            f"Use linear interpolation. Round to {self.config.decimal_places} decimal places."
        )
        return {"question": question, "answer": answer, "task_type": "percentile"}

    def _make_z_score(self, rng: random.Random) -> dict:
        mean = rng.randint(50, 150)
        std = rng.randint(5, 30)
        x = mean + rng.randint(-3, 3) * std + rng.randint(-std, std)
        z = (x - mean) / std
        answer = self._fmt(z)
        question = (
            f"A dataset has a mean of {mean} and a standard deviation of {std}. "
            f"What is the z-score of the value {x}? "
            f"Round your answer to {self.config.decimal_places} decimal places."
        )
        return {"question": question, "answer": answer, "task_type": "z_score"}

    def __getitem__(self, idx: int) -> dict:
        rng = random.Random(self.seed + idx)
        task_type = rng.choices(self.config.task_types, weights=self.config.task_weights, k=1)[0]

        generators = {
            "mean": self._make_mean,
            "median": self._make_median,
            "mode": self._make_mode,
            "weighted_mean": self._make_weighted_mean,
            "std_dev": self._make_std_dev,
            "percentile": self._make_percentile,
            "z_score": self._make_z_score,
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
                    "min_data_size": self.config.min_data_size,
                    "max_data_size": self.config.max_data_size,
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
            a_parts = [float(x.strip()) for x in answer.split(",")]
            o_parts = [float(x.strip()) for x in oracle.split(",")]
            if len(a_parts) != len(o_parts):
                return 0.0
            max_err = max(abs(a - o) for a, o in zip(a_parts, o_parts))
            if max_err < 10 ** (-(self.config.decimal_places)):
                return 1.0
            if max_err < 0.1:
                return 0.5
            return 0.0
        except (ValueError, TypeError):
            return 0.0


class DescriptiveStatsCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(DescriptiveStatsCurriculum.__name__, DescriptiveStatsConfig)
        self._define_attributes(
            RangeAttributeDefinition(
                name="data_size",
                levels=[5, 10, 20, 50],
                lower_field_name="min_data_size",
                upper_field_name="max_data_size",
                description="Size of data sets",
            ),
            ScalarAttributeDefinition(
                name="decimal_places",
                field_name="decimal_places",
                levels=[1, 2, 3, 4],
                description="Decimal precision required",
            ),
        )


register_dataset(DATASET_NAME, DescriptiveStatsDataset, DescriptiveStatsConfig, DescriptiveStatsCurriculum)
