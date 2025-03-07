"""Find the largest island in a grid of 1s and 0s.

A popular Leetcode problem:
https://leetcode.com/problems/max-area-of-island/description/
"""

from collections import deque
from dataclasses import dataclass
from random import Random
from typing import Optional

from ..coaching import AttributeType, BaseCurriculum, RangeAttributeDefinition
from ..factory import ProceduralDataset, register_dataset

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

    min_rows: int = 5  # Minimum number of rows in the grid
    max_rows: int = 10  # Maximum number of rows in the grid
    min_cols: int = 5  # Minimum number of columns in the grid
    max_cols: int = 10  # Maximum number of columns in the grid
    min_num_islands: int = 0
    max_num_islands: int = (
        5  # Maximum number of islands (actual max might be smaller due to merging of islands during random walk)
    )
    min_island_size: int = 0
    max_island_size: int = (
        10  # Maximum size of an island (actual max might be larger due to merging of islands during random walk)
    )

    size: int = 500  # Virtual dataset size
    seed: Optional[int] = None

    def validate(self):
        """Validate configuration parameters"""
        assert 1 <= self.min_rows <= self.max_rows, "Invalid rows range"
        assert 1 <= self.min_cols <= self.max_cols, "Invalid cols range"
        assert 0 <= self.min_num_islands <= self.max_num_islands, "Invalid num_islands range"
        assert 0 <= self.min_island_size <= self.max_island_size, "Invalid island_size range"


class LargestIslandDataset(ProceduralDataset):
    """Generates Largest Island exercises with configurable difficulty"""

    def __init__(self, config: LargestIslandConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _is_valid_cell(self, r: int, c: int, rows: int, cols: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols

    def _create_grid(self, rng: Random, rows: int, cols: int, num_islands: int) -> list[list[int]]:
        """Create a random grid of islands using a random walk algorithm"""
        grid = [[0] * cols for _ in range(rows)]
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

        def create_island():
            r, c = rng.randint(0, rows - 1), rng.randint(0, cols - 1)
            capped_size = min(rng.randint(self.config.min_island_size, self.config.max_island_size), rows * cols)
            for _ in range(capped_size):
                grid[r][c] = 1
                rng.shuffle(directions)
                for dr, dc in directions:
                    new_r, new_c = r + dr, c + dc
                    if self._is_valid_cell(new_r, new_c, rows, cols) and grid[new_r][new_c] == 0:
                        r, c = new_r, new_c
                        break

        for _ in range(num_islands):
            create_island()

        return grid

    def _get_largest_island(self, grid: list[list[int]]) -> int:
        """Find the largest island in the grid"""
        rows, cols = len(grid), len(grid[0])
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
                    if (
                        self._is_valid_cell(new_r, new_c, rows, cols)
                        and (new_r, new_c) not in visited
                        and grid[new_r][new_c] == 1
                    ):
                        area += 1
                        visited.add((new_r, new_c))
                        queue.append((new_r, new_c))
            return area

        max_area = 0
        for r in range(rows):
            for c in range(cols):
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

        rows = rng.randint(self.config.min_rows, self.config.max_rows)
        cols = rng.randint(self.config.min_cols, self.config.max_cols)
        num_islands = rng.randint(self.config.min_num_islands, self.config.max_num_islands)
        grid = self._create_grid(rng, rows, cols, num_islands)
        grid_str = self._grid_to_string(grid)

        answer = self._get_largest_island(grid)

        return {
            "question": QUESTION_TEMPLATE.format(rows=rows, cols=cols, grid=grid_str),
            "answer": str(answer),
            "metadata": {
                "grid": grid,
                "solution": answer,
                "difficulty": {
                    "rows": rows,
                    "cols": cols,
                    "num_islands": num_islands,
                },
            },
        }


class LargestIslandCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(LargestIslandCurriculum.__name__, LargestIslandConfig)

        # Define attributes
        self._define_attributes(
            RangeAttributeDefinition(
                name="rows",
                levels=[5, 10, 50, 100],
                default_level=0,
                description="Number of rows in the grid",
                attr_type=AttributeType.APPEND,
                min_value=1,
                lower_field_name="min_rows",
                upper_field_name="max_rows",
            ),
            RangeAttributeDefinition(
                name="cols",
                levels=[5, 10, 50, 100],
                default_level=0,
                description="Number of columns in the grid",
                attr_type=AttributeType.APPEND,
                min_value=1,
                lower_field_name="min_cols",
                upper_field_name="max_cols",
            ),
            RangeAttributeDefinition(
                name="num_islands",
                levels=[2, 5, 10, 20],
                default_level=0,
                description="Number of islands in the grid",
                attr_type=AttributeType.APPEND,
                min_value=0,
                lower_field_name="min_num_islands",
                upper_field_name="max_num_islands",
            ),
            RangeAttributeDefinition(
                name="island_size",
                levels=[5, 10, 20, 30],
                default_level=0,
                description="Size of the islands in the grid",
                attr_type=AttributeType.APPEND,
                min_value=0,
                lower_field_name="min_island_size",
                upper_field_name="max_island_size",
            ),
        )


register_dataset("largest_island", LargestIslandDataset, LargestIslandConfig, LargestIslandCurriculum)
