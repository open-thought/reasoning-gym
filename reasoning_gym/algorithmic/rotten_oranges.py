"""Find how many steps it takes for all oranges in a grid to rot.

A popular Leetcode problem:
https://leetcode.com/problems/rotting-oranges/description/
"""

from collections import deque
from dataclasses import dataclass
from random import Random
from typing import Optional

from ..factory import ProceduralDataset, register_dataset

QUESTION_TEMPLATE = """You are given an n x n grid where each cell can have one of three values:
- 0 representing an empty cell
- 1 representing a fresh orange
- 2 representing a rotten orange

Every minute, any fresh orange that is 4-directionally adjacent to a rotten orange becomes rotten.

Your task is determine the minimum number of minutes that must elapse until no cell has a fresh orange.
If this is impossible, return -1.

Example:
- Input: Determine the minimum number of minutes that must elapse until no cell in the grid below has a fresh orange:
    2 1 1
    1 1 0
    0 1 1
- Output: 4

Now, determine the minimum number of minutes that must elapse until no cell in the grid below has a fresh orange:
{matrix}
"""


@dataclass
class RottenOrangesConfig:
    """Configuration for Rotten Oranges dataset generation"""

    min_n: int = 10  # Minimum size of the matrix
    max_n: int = 30  # Maximum size of the matrix
    p_oranges: float = 0.85  # Percent of grid cells populated with oranges
    p_rotten: float = 0.1  # Percent of oranges that are initially rotten

    size: int = 500  # Virtual dataset size
    seed: Optional[int] = None

    def validate(self):
        """Validate configuration parameters"""
        assert 1 <= self.min_n, "min_n must be at least 1"
        assert self.min_n <= self.max_n, "min_n must be less than or equal to max_n"
        assert 0 < self.p_oranges <= 1, "p_oranges must be between 0 and 1"
        assert 0 < self.p_rotten <= 1, "p_rotten must be between 0 and 1"


class RottenOrangesDataset(ProceduralDataset):
    """Generates Rotten Oranges exercises with configurable difficulty"""

    def __init__(self, config: RottenOrangesConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _matrix_to_str(self, matrix: list[list[int]]) -> str:
        """Get a string representation of the matrix"""
        return "\n".join(" ".join(str(x) for x in row) for row in matrix)

    def _get_initial_matrix(self, rng: Random) -> list[list[int]]:
        """Generate a random matrix with oranges"""
        n = rng.randint(self.config.min_n, self.config.max_n)
        matrix = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if rng.random() < self.config.p_oranges:
                    matrix[i][j] = 1
                    if rng.random() < self.config.p_rotten:
                        matrix[i][j] = 2
        return matrix

    def _get_answer(self, matrix: list[list[int]]) -> int:
        """Calculate the number of steps it takes for all oranges to rot"""
        ROWS, COLS = len(matrix), len(matrix[0])
        DIRS = [[1, 0], [-1, 0], [0, 1], [0, -1]]

        q, visited = deque(), set()
        infected, healthy, clock = 0, 0, 0

        for r in range(ROWS):
            for c in range(COLS):
                if matrix[r][c] == 2:
                    visited.add((r, c))
                    q.append((r, c))
                elif matrix[r][c] == 1:
                    healthy += 1

        while True:
            temp = deque()
            while q:
                r, c = q.popleft()
                for dr, dc in DIRS:
                    new_r, new_c = r + dr, c + dc
                    if (
                        0 <= new_r < ROWS
                        and 0 <= new_c < COLS
                        and (new_r, new_c) not in visited
                        and matrix[new_r][new_c] == 1
                    ):
                        infected += 1
                        visited.add((new_r, new_c))
                        temp.append((new_r, new_c))
            if temp:
                q = temp
            else:
                break
            clock += 1

        return clock if infected == healthy else -1

    def __getitem__(self, idx: int) -> dict:
        """Generate a single Rotten Oranges question"""
        rng = Random(self.seed + idx)

        matrix = self._get_initial_matrix(rng)
        matrix_str = self._matrix_to_str(matrix)
        answer = self._get_answer(matrix)

        return {
            "question": QUESTION_TEMPLATE.format(matrix=matrix_str),
            "answer": str(answer),
            "metadata": {"matrix": matrix, "solution": answer},
        }


register_dataset("rotten_oranges", RottenOrangesDataset, RottenOrangesConfig)
