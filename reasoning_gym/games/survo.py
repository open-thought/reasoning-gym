"""
Survo dataset, adapted for Reasoning Gym from SynthRL: https://github.com/MiniMax-AI/SynLogic/tree/main/games/tasks/survo
"""

# TODO: add unit tests

from dataclasses import dataclass
from random import Random
from typing import Any, Optional

import numpy as np

from reasoning_gym.dataset import ProceduralDataset

PROMPT_TEMPLATES = [
    "Given a {n}*{n} matrix where the last element of each row and column equals the sum of the other elements in that row or column. The matrix is:\n{matrix}\nwhere some elements are replaced with X. You have a set of numbers {numbers} that can be filled into the X positions to satisfy the rules. Please fill in the matrix. Each number can only be used once.",
    "You have a {n}*{n} matrix with some positions already filled with numbers and others marked with X. The matrix is:\n{matrix}\nThe last number in each row and column represents the sum of all other numbers in that row or column. You need to fill in the X positions using the numbers {numbers} to satisfy these conditions. Each number can only be used once.",
    "Complete the following Survo puzzle. In this {n}*{n} matrix:\n{matrix}\nthe cells marked with X need to be filled with numbers. The last number in each row and column equals the sum of all other numbers in that row or column. You can use the following numbers: {numbers}. Each number can only be used once.",
    "In this {n}*{n} Survo matrix puzzle:\n{matrix}\nthe X cells need to be filled with numbers from the set {numbers}. The last element in each row and column is the sum of all other elements in that row or column. Each number can only be used once. Provide the completed matrix.",
    "Solve this {n}*{n} matrix puzzle:\n{matrix}\nwhere X represents empty cells that need to be filled. The last number in each row and column equals the sum of all other numbers in that row or column. You have the numbers {numbers} to place in the empty cells. Each number can only be used once.",
]


@dataclass
class SurvoConfig:
    # TODO: revisit parameters
    n: int = 4
    x: int = 3
    min_num: int = 1
    max_num: int = 9
    seed: Optional[int] = None
    size: int = 500  # Virtual dataset size

    def validate(self):
        """Validate configuration parameters"""
        assert self.n > 3, "n must be greater than 3"
        assert self.x > 0 and self.x <= (self.n - 1) * (
            self.n - 1
        ), f"x must be > 0 and <= {(self.n - 1) * (self.n - 1)}"
        assert self.min_num < self.max_num, "min_num must be less than max_num"


class SurvoDataset(ProceduralDataset):
    def __init__(self, config: SurvoConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def __len__(self) -> int:
        return self.config.size

    def __iter__(self):
        self._current_idx = 0
        return self

    def __next__(self):
        if self._current_idx >= self.config.size:
            raise StopIteration
        item = self[self._current_idx]
        self._current_idx += 1
        return item

    def __getitem__(self, idx: int) -> dict:
        rng = Random(self.config.seed + idx)

        matrix, original_matrix, candidate_numbers = self._generate_valid_matrix(
            rng, self.config.n, self.config.x, self.config.min_num, self.config.max_num
        )

        question = rng.choice(PROMPT_TEMPLATES).format(
            n=self.config.n, matrix=original_matrix, numbers=candidate_numbers
        )

        return {
            "question": question,
            "answer": str(matrix.tolist()),
            "metadata": {
                "original_matrix": original_matrix.tolist(),
                "filled_matrix": matrix.tolist(),
                "candidate_numbers": candidate_numbers,
                "n": self.config.n,
                "x": self.config.x,
                "min_num": self.config.min_num,
                "max_num": self.config.max_num,
            },
        }

    def _generate_valid_matrix(
        self, rng: Random, n: int, x: int, min_num: int, max_num: int
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        matrix = np.zeros((n, n), dtype=int)

        for i in range(n - 1):
            for j in range(n - 1):
                matrix[i, j] = rng.randint(min_num, max_num)

        for i in range(n - 1):
            row_sum = sum(matrix[i, 0 : n - 1])
            matrix[i, n - 1] = row_sum

            col_sum = sum(matrix[0 : n - 1, i])
            matrix[n - 1, i] = col_sum

        matrix[n - 1, n - 1] = sum(matrix[0 : n - 1, n - 1])

        filled_matrix = matrix.copy()

        positions = [(i, j) for i in range(n - 1) for j in range(n - 1)]
        selected_positions = rng.sample(positions, x)

        candidate_numbers = []
        for i, j in selected_positions:
            candidate_numbers.append(int(matrix[i, j]))
            matrix[i, j] = 0

        return filled_matrix, matrix, candidate_numbers

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        if not isinstance(answer, str):
            return 0.0

        n = entry["metadata"]["n"]
        grid = self._parse_grid(answer)
        true_grid = entry["metadata"]["filled_matrix"]

        if len(grid) != n or any(len(row) != n for row in grid):
            return 0.0

        for i in range(n):
            for j in range(n):
                if grid[i][j] != true_grid[i][j]:
                    return 0.0

        return 1.0

    def _parse_grid(self, answer: str) -> list[list[str]]:
        grid = []
        for line in answer.strip().split("\n"):
            for c in line.strip().split():
                try:
                    grid.append([int(c)])
                except ValueError:
                    continue  # Ignore non-integer values
        return grid


# TODO: add curriculum
