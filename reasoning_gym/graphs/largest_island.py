"""Find the largest island in a grid of 1s and 0s.

A popular Leetcode problem:
https://leetcode.com/problems/max-area-of-island/description/
"""

from collections import deque
from dataclasses import dataclass
from random import Random
from typing import Optional

from ..factory import ProceduralDataset, register_dataset

MIN_MAP_DIM = 1

QUESTION_TEMPLATE = """You are given the following {rows} x {cols} binary matrix grid:
{grid}

An island is a group of 1's (representing land) connected 4-directionally (horizontal or vertical).
You may assume all four edges of the grid are surrounded by water.

The area of an island is the number of cells with a value 1 in the island.

Return the maximum area of an island in grid. If there is no island, return 0.
"""


@dataclass
class LargestIslandConfig:
    """Configuration for Largest Island dataset generation"""

    rows: int = 10  # Number of rows in the grid
    cols: int = 10  # Number of columns in the grid
    max_num_islands: int = (
        5  # Maximum number of islands (actual max might be smaller due to merging of islands during random walk)
    )
    max_island_size: int = (
        10  # Maximum size of an island (actual max might be larger due to merging of islands during random walk)
    )

    size: int = 500  # Virtual dataset size
    seed: Optional[int] = None

    def validate(self):
        """Validate configuration parameters"""
        assert MIN_MAP_DIM <= self.rows, f"rows must be between larger than {MIN_MAP_DIM}"
        assert MIN_MAP_DIM <= self.cols, f"cols must be between larger than {MIN_MAP_DIM}"
        assert 0 <= self.max_num_islands, "max_num_islands must be non-negative"
        assert 0 <= self.max_island_size, "max_island_size must be non-negative"


class LargestIslandDataset(ProceduralDataset):
    """Generates Largest Island exercises with configurable difficulty"""

    def __init__(self, config: LargestIslandConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _is_valid_cell(self, r: int, c: int) -> bool:
        return 0 <= r < self.config.rows and 0 <= c < self.config.cols

    def _create_grid(self, rng: Random) -> list[list[int]]:
        """Create a random grid of islands using a random walk algorithm"""
        grid = [[0] * self.config.cols for _ in range(self.config.rows)]
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

        def create_island():
            r, c = rng.randint(0, self.config.rows - 1), rng.randint(0, self.config.cols - 1)
            capped_size = min(rng.randint(0, self.config.max_island_size), self.config.rows * self.config.cols)
            for _ in range(capped_size):
                grid[r][c] = 1
                rng.shuffle(directions)
                for dr, dc in directions:
                    new_r, new_c = r + dr, c + dc
                    if self._is_valid_cell(new_r, new_c) and grid[new_r][new_c] == 0:
                        r, c = new_r, new_c
                        break

        num_islands = rng.randint(0, self.config.max_num_islands)
        for _ in range(num_islands):
            create_island()

        return grid

    def _get_largest_island(self, grid: list[list[int]]) -> int:
        """Find the largest island in the grid"""
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        visited = set()

        def bfs(r, c):
            area = 1
            visited.add((r, c))
            queue = deque([(r, c)])
            while queue:
                r, c = queue.popleft()
                for dr, dc in directions:
                    new_r, new_c = r + dr, c + dc
                    if self._is_valid_cell(new_r, new_c) and (new_r, new_c) not in visited and grid[new_r][new_c] == 1:
                        area += 1
                        visited.add((new_r, new_c))
                        queue.append((new_r, new_c))
            return area

        max_area = 0
        for r in range(self.config.rows):
            for c in range(self.config.cols):
                if grid[r][c] == 1 and (r, c) not in visited:
                    max_area = max(max_area, bfs(r, c))

        return max_area

    def _grid_to_string(self, grid: list[list[int]]) -> str:
        """Convert grid to a string representation"""
        return "\n".join(" ".join(str(cell) for cell in row) for row in grid)

    def _string_to_board(self, grid_str: str) -> list[list[int]]:
        """Convert string representation to a grid"""
        return [[int(cell) for cell in row.split()] for row in grid_str.split("\n")]

    def __getitem__(self, idx: int) -> dict:
        """Generate a single Largest Island question"""
        rng = Random(self.seed + idx)

        grid = self._create_grid(rng)
        grid_str = self._grid_to_string(grid)

        answer = self._get_largest_island(grid)

        return {
            "question": QUESTION_TEMPLATE.format(rows=self.config.rows, cols=self.config.cols, grid=grid_str),
            "answer": str(answer),
            "metadata": {"grid": grid, "solution": answer},
        }


register_dataset("largest_island", LargestIslandDataset, LargestIslandConfig)
