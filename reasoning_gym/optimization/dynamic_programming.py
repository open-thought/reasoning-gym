import random
from dataclasses import dataclass, field
from typing import Any, Optional

from ..coaching import BaseCurriculum, ScalarAttributeDefinition
from ..factory import ProceduralDataset, register_dataset

DATASET_NAME = "dynamic_programming"

TASK_TYPES = ("lcs", "coin_change", "lis", "edit_distance", "staircase")


def _lcs_length(s1: str, s2: str) -> tuple[int, str]:
    m, n = len(s1), len(s2)
    dp = [[""] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + s1[i - 1]
            else:
                dp[i][j] = dp[i - 1][j] if len(dp[i - 1][j]) >= len(dp[i][j - 1]) else dp[i][j - 1]
    return len(dp[m][n]), dp[m][n]


def _coin_change(coins: list[int], amount: int) -> int:
    dp = [float("inf")] * (amount + 1)
    dp[0] = 0
    for c in coins:
        for a in range(c, amount + 1):
            dp[a] = min(dp[a], dp[a - c] + 1)
    return dp[amount] if dp[amount] != float("inf") else -1


def _lis_length(arr: list[int]) -> int:
    if not arr:
        return 0
    dp = [1] * len(arr)
    for i in range(1, len(arr)):
        for j in range(i):
            if arr[j] < arr[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)


def _edit_distance(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


@dataclass
class DynamicProgrammingConfig:
    min_str_len: int = 4
    max_str_len: int = 8
    min_arr_len: int = 5
    max_arr_len: int = 10
    max_staircase: int = 15
    task_types: tuple[str, ...] = TASK_TYPES
    task_weights: list[float] = field(default_factory=lambda: [0.2, 0.2, 0.2, 0.2, 0.2])
    seed: Optional[int] = None
    size: int = 500

    def validate(self) -> None:
        assert self.size > 0, "size must be positive"
        assert self.min_str_len >= 2, "min_str_len must be >= 2"
        assert self.max_str_len >= self.min_str_len, "max_str_len must be >= min_str_len"
        assert self.min_arr_len >= 3, "min_arr_len must be >= 3"
        assert self.max_arr_len >= self.min_arr_len, "max_arr_len must be >= min_arr_len"
        assert self.max_staircase >= 2, "max_staircase must be >= 2"
        assert len(self.task_types) > 0, "must have at least one task type"
        assert all(t in TASK_TYPES for t in self.task_types), f"invalid task type"
        assert len(self.task_weights) == len(self.task_types), "weights must match types"


class DynamicProgrammingDataset(ProceduralDataset):
    def __init__(self, config: DynamicProgrammingConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _make_lcs(self, rng: random.Random) -> dict:
        l1 = rng.randint(self.config.min_str_len, self.config.max_str_len)
        l2 = rng.randint(self.config.min_str_len, self.config.max_str_len)
        chars = "ABCDEFGH"
        s1 = "".join(rng.choice(chars) for _ in range(l1))
        s2 = "".join(rng.choice(chars) for _ in range(l2))
        length, _ = _lcs_length(s1, s2)
        question = (
            f"Find the length of the longest common subsequence (LCS) of '{s1}' and '{s2}'. "
            f"Give your answer as a single integer."
        )
        return {"question": question, "answer": str(length), "task_type": "lcs"}

    def _make_coin_change(self, rng: random.Random) -> dict:
        num_coins = rng.randint(2, 4)
        coins = sorted(set(rng.randint(1, 10) for _ in range(num_coins + 2)))[:num_coins]
        if 1 not in coins:
            coins = [1] + coins
        amount = rng.randint(5, 25)
        result = _coin_change(coins, amount)
        question = (
            f"What is the minimum number of coins needed to make {amount} "
            f"using coins of denominations {coins}? Each denomination can be used unlimited times. "
            f"Give your answer as a single integer."
        )
        return {"question": question, "answer": str(result), "task_type": "coin_change"}

    def _make_lis(self, rng: random.Random) -> dict:
        n = rng.randint(self.config.min_arr_len, self.config.max_arr_len)
        arr = [rng.randint(1, 50) for _ in range(n)]
        length = _lis_length(arr)
        question = (
            f"Find the length of the longest strictly increasing subsequence in {arr}. "
            f"Give your answer as a single integer."
        )
        return {"question": question, "answer": str(length), "task_type": "lis"}

    def _make_edit_distance(self, rng: random.Random) -> dict:
        l1 = rng.randint(self.config.min_str_len, self.config.max_str_len)
        l2 = rng.randint(self.config.min_str_len, self.config.max_str_len)
        chars = "abcdefgh"
        s1 = "".join(rng.choice(chars) for _ in range(l1))
        s2 = "".join(rng.choice(chars) for _ in range(l2))
        dist = _edit_distance(s1, s2)
        question = (
            f"What is the minimum edit distance (Levenshtein distance) between "
            f"'{s1}' and '{s2}'? Operations: insert, delete, or substitute a character. "
            f"Give your answer as a single integer."
        )
        return {"question": question, "answer": str(dist), "task_type": "edit_distance"}

    def _make_staircase(self, rng: random.Random) -> dict:
        n = rng.randint(3, self.config.max_staircase)
        ways = [0] * (n + 1)
        ways[0] = 1
        ways[1] = 1
        for i in range(2, n + 1):
            ways[i] = ways[i - 1] + ways[i - 2]
        question = (
            f"You are climbing a staircase with {n} steps. Each time you can climb 1 or 2 steps. "
            f"How many distinct ways can you reach the top? Give your answer as a single integer."
        )
        return {"question": question, "answer": str(ways[n]), "task_type": "staircase"}

    def __getitem__(self, idx: int) -> dict:
        rng = random.Random(self.seed + idx)
        task_type = rng.choices(self.config.task_types, weights=self.config.task_weights, k=1)[0]

        generators = {
            "lcs": self._make_lcs,
            "coin_change": self._make_coin_change,
            "lis": self._make_lis,
            "edit_distance": self._make_edit_distance,
            "staircase": self._make_staircase,
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
                    "max_str_len": self.config.max_str_len,
                    "max_arr_len": self.config.max_arr_len,
                },
            },
        }


class DynamicProgrammingCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(DynamicProgrammingCurriculum.__name__, DynamicProgrammingConfig)
        self._define_attributes(
            ScalarAttributeDefinition(
                name="max_str_len",
                field_name="max_str_len",
                levels=[5, 8, 12, 15],
                description="Maximum string length",
            ),
            ScalarAttributeDefinition(
                name="max_arr_len",
                field_name="max_arr_len",
                levels=[5, 10, 15, 20],
                description="Maximum array length",
            ),
        )


register_dataset(DATASET_NAME, DynamicProgrammingDataset, DynamicProgrammingConfig, DynamicProgrammingCurriculum)
