"""Find the shortest path between a start and end point in a grid"""

from collections import deque
from dataclasses import dataclass
from random import Random
from typing import Optional

from ..factory import ProceduralDataset, register_dataset

QUESTION_TEMPLATE = """Your task is to find the length of the shortest path from the start to the destination point in a grid.

The grid is represented as a matrix with the following types of cells:
- *: your starting point
- #: your destination point
- O: an open cell
- X: a blocked cell

Therefore, you need to find the length of the shortest path from * to #, moving only through open cells.
If there is no path from * to #, return -1.

Example:
- Input: Find the length of the shortest path from * to # in the following grid:
    X X X X X
    X * O O X
    X O X O X
    X X X O #
- Output: 5

Now, find the length of the shortest path from * to # in the following grid:
{grid}
"""


@dataclass
class ShortestPathConfig:
    """Configuration for Shortest Path dataset generation"""

    min_rows: int = 10
    max_rows: int = 30
    min_cols: int = 10
    max_cols: int = 30
    p_blocked: float = 0.4

    size: int = 500  # Virtual dataset size
    seed: Optional[int] = None

    def validate(self):
        """Validate configuration parameters"""
        assert 1 <= self.min_rows, "min_rows must be at least 1"
        assert self.min_rows <= self.max_rows, "min_rows must be less than or equal to max_rows"
        assert 1 <= self.min_cols, "min_cols must be at least 1"
        assert self.min_cols <= self.max_cols, "min_cols must be less than or equal to max_cols"
        assert 0 <= self.p_blocked <= 1, "p_blocked must be between 0 and 1"


class ShortestPathDataset(ProceduralDataset):
    """Generates Shortest Path exercises with configurable difficulty"""

    def __init__(self, config: ShortestPathConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _get_grid(self, rng: Random) -> list[list[str]]:
        """Generate a random grid with open and blocked cells"""

        rows, cols = rng.randint(self.config.min_rows, self.config.max_rows), rng.randint(
            self.config.min_cols, self.config.max_cols
        )
        grid = [["X" if rng.random() < self.config.p_blocked else "O" for _ in range(cols)] for _ in range(rows)]

        start_r, start_c = rng.randint(0, rows - 1), rng.randint(0, cols - 1)
        grid[start_r][start_c] = "*"

        while True:
            end_r, end_c = rng.randint(0, rows - 1), rng.randint(0, cols - 1)
            if (end_r, end_c) != (start_r, start_c):
                grid[end_r][end_c] = "#"
                break

        return grid

    def _matrix_to_str(self, matrix: list[list[int]]) -> str:
        """Get a string representation of the matrix"""
        return "\n".join(" ".join(str(x) for x in row) for row in matrix)

    def _get_answer(self, matrix: list[list[str]]) -> int:
        """Run BFS to find the shortest path length"""
        ROWS, COLS = len(matrix), len(matrix[0])
        DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        start_r, start_c = next((r, c) for r in range(ROWS) for c in range(COLS) if matrix[r][c] == "*")
        queue = deque([(start_r, start_c)])
        steps = 0

        while queue:
            steps += 1
            for _ in range(len(queue)):
                r, c = queue.popleft()
                for dr, dc in DIRS:
                    new_r, new_c = r + dr, c + dc
                    if 0 <= new_r < ROWS and 0 <= new_c < COLS:
                        if matrix[new_r][new_c] == "#":
                            return steps
                        if matrix[new_r][new_c] == "O":
                            matrix[new_r][new_c] = "X"
                            queue.append((new_r, new_c))

        return -1

    def __getitem__(self, idx: int) -> dict:
        """Generate a single Shortest Path question"""
        rng = Random(self.seed + idx)

        matrix = self._get_grid(rng)
        matrix_str = self._matrix_to_str(matrix)
        answer = self._get_answer(matrix)

        return {
            "question": QUESTION_TEMPLATE.format(grid=matrix_str),
            "answer": str(answer),
            "metadata": {"matrix": matrix, "solution": answer},
        }


register_dataset("shortest_path", ShortestPathDataset, ShortestPathConfig)
