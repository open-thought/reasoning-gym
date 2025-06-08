"""
Survo dataset, adapted for Reasoning Gym from SynthRL: https://github.com/MiniMax-AI/SynLogic/tree/main/games/tasks/survo
"""

from dataclasses import dataclass
from random import Random
from typing import Any, Optional

import numpy as np

from ..coaching import BaseCurriculum, RangeAttributeDefinition
from ..factory import ProceduralDataset, register_dataset

DATASET_NAME = "survo"

PROMPT_TEMPLATES = [
    "Given a {n}*{n} matrix where the last element of each row and column equals the sum of the other elements in that row or column. The matrix is:\n{matrix}\nwhere some elements are replaced with 0. You have a set of numbers {numbers} that can be filled into the 0 positions to satisfy the rules. Please fill in the matrix. Each number can only be used once.",
    "You have a {n}*{n} matrix with some positions already filled with numbers and others marked with 0. The matrix is:\n{matrix}\nThe last number in each row and column represents the sum of all other numbers in that row or column. You need to fill in the 0 positions using the numbers {numbers} to satisfy these conditions. Each number can only be used once.",
    "Complete the following Survo puzzle. In this {n}*{n} matrix:\n{matrix}\nthe cells marked with 0 need to be filled with numbers. The last number in each row and column equals the sum of all other numbers in that row or column. You can use the following numbers: {numbers}. Each number can only be used once.",
    "In this {n}*{n} Survo matrix puzzle:\n{matrix}\nthe 0 cells need to be filled with numbers from the set {numbers}. The last element in each row and column is the sum of all other elements in that row or column. Each number can only be used once. Provide the completed matrix.",
    "Solve this {n}*{n} matrix puzzle:\n{matrix}\nwhere 0 represents empty cells that need to be filled. The last number in each row and column equals the sum of all other numbers in that row or column. You have the numbers {numbers} to place in the empty cells. Each number can only be used once.",
]


@dataclass
class SurvoConfig:
    min_board_size: int = 4
    max_board_size: int = 5
    min_empty: int = 3
    max_empty: int = 5
    min_num: int = 1
    max_num: int = 9
    seed: Optional[int] = None
    size: int = 500  # Virtual dataset size

    def validate(self):
        """Validate configuration parameters"""
        assert self.min_board_size > 3, "min_board_size must be greater than 3"
        assert self.max_board_size >= self.min_board_size, "max_board_size must be >= min_board_size"
        assert self.min_empty > 0, "min_empty must be > 0"
        assert self.max_empty <= (self.min_board_size - 1) * (
            self.min_board_size - 1
        ), f"max_empty must be <= {(self.min_board_size - 1) * (self.min_board_size - 1)}"
        assert self.min_empty <= self.max_empty, "min_empty must be <= max_empty"
        assert self.min_num > 0, "min_num must be > 0"
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

        board_size = rng.randint(self.config.min_board_size, self.config.max_board_size)
        num_empty = rng.randint(self.config.min_empty, self.config.max_empty)

        filled_matrix, puzzle, candidate_numbers = self._generate_valid_matrix(
            rng, board_size, num_empty, self.config.min_num, self.config.max_num
        )

        puzzle_str = "\n".join(" ".join(str(x) for x in row) for row in puzzle)
        solution_str = "\n".join(" ".join(str(x) for x in row) for row in filled_matrix)

        question = rng.choice(PROMPT_TEMPLATES).format(n=board_size, matrix=puzzle_str, numbers=candidate_numbers)

        return {
            "question": question,
            "answer": solution_str,
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_idx": idx,
                "puzzle": puzzle.tolist(),
                "solution": filled_matrix.tolist(),
                "candidate_numbers": candidate_numbers,
                "board_size": board_size,
                "num_empty": num_empty,
                "min_num": self.config.min_num,
                "max_num": self.config.max_num,
                "difficulty": {
                    "board_size": (self.config.min_board_size, self.config.max_board_size),
                    "empty": (self.config.min_empty, self.config.max_empty),
                },
            },
        }

    def _generate_valid_matrix(
        self, rng: Random, n: int, num_empty: int, min_num: int, max_num: int
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
        selected_positions = rng.sample(positions, num_empty)

        candidate_numbers = []
        for i, j in selected_positions:
            candidate_numbers.append(int(matrix[i, j]))
            matrix[i, j] = 0

        return filled_matrix, matrix, candidate_numbers

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        if not isinstance(answer, str):
            return 0.0

        board_size = entry["metadata"]["board_size"]
        grid = self._parse_grid(answer)
        true_grid = entry["metadata"]["solution"]

        if len(grid) != board_size or any(len(row) != board_size for row in grid):
            return 0.0

        for i in range(board_size):
            for j in range(board_size):
                if grid[i][j] != true_grid[i][j]:
                    return 0.0

        return 1.0

    def _parse_grid(self, answer: str) -> list[list[str]]:
        grid = []
        for line in answer.strip().split("\n"):
            row = []
            for c in line.strip().split():
                try:
                    row.append(int(c))
                except ValueError:
                    continue  # Ignore non-integer values
            if row:
                grid.append(row)
        return grid


class SurvoCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(SurvoCurriculum.__name__, SurvoConfig)

        self._define_attributes(
            RangeAttributeDefinition(
                name="board_size",
                levels=[4, 5, 6, 7],
                description="Board size (n x n)",
                lower_field_name="min_board_size",
                upper_field_name="max_board_size",
            ),
            RangeAttributeDefinition(
                name="empty",
                levels=[4, 9, 16, 25],
                description="Number of empty cells",
                lower_field_name="min_empty",
                upper_field_name="max_empty",
            ),
        )


register_dataset(DATASET_NAME, SurvoDataset, SurvoConfig, SurvoCurriculum)
