"""Find the longest simple path between two points in a grid."""

from dataclasses import dataclass
from functools import lru_cache
from random import Random
from typing import Any, Optional

from ..factory import ProceduralDataset, register_dataset

DATASET_NAME = "longest_path"

QUESTION_TEMPLATE = """Your task is to find the length of the longest simple path from the start to the destination in a grid.

The grid uses these symbols:
- *: start
- #: destination
- O: open cell
- X: blocked cell

You may move only up, down, left, or right, and you may not move through blocked cells.
A valid path starts at * and ends at #. You must never visit the same cell more than once.
The length of a path is the number of moves between cells, not the number of cells visited.

What is the maximum possible path length? Output only the non-negative integer, with no explanation.

Grid:
{grid}
"""


@dataclass
class LongestPathConfig:
    """Configuration for Longest Path grid task generation.

    Exact longest-simple-path search is exponential in the number of open cells.
    ``max_open_cells`` deliberately caps that number to keep generation tractable.
    """

    min_rows: int = 3
    max_rows: int = 5
    min_cols: int = 3
    max_cols: int = 5
    p_blocked: float = 0.25
    max_open_cells: int = 18
    size: int = 500
    seed: Optional[int] = None

    def validate(self) -> None:
        """Validate generation parameters and enforce a solver-size cap."""
        assert self.min_rows >= 2, "min_rows must be at least 2"
        assert self.min_rows <= self.max_rows, "min_rows must be <= max_rows"
        assert self.min_cols >= 2, "min_cols must be at least 2"
        assert self.min_cols <= self.max_cols, "min_cols must be <= max_cols"
        assert self.max_rows <= 10, "max_rows must be <= 10"
        assert self.max_cols <= 10, "max_cols must be <= 10"
        assert 0.0 <= self.p_blocked <= 1.0, "p_blocked must be between 0 and 1"
        assert self.max_open_cells <= 18, "max_open_cells must be <= 18 for exact search"
        # The longest carved Manhattan route across the configured grid has this many cells.
        assert self.max_open_cells >= self.max_rows + self.max_cols - 1, (
            "max_open_cells must be at least max_rows + max_cols - 1 "
            "so every generated grid can retain a guaranteed route"
        )
        assert self.size > 0, "size must be positive"


class LongestPathDataset(ProceduralDataset):
    """Procedurally generates small grids and their longest simple-path lengths."""

    def __init__(self, config: LongestPathConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    @staticmethod
    def _matrix_to_str(matrix: list[list[str]]) -> str:
        """Render the grid as space-separated cell symbols."""
        return "\n".join(" ".join(row) for row in matrix)

    def _get_grid(self, rng: Random, rows: int, cols: int) -> list[list[str]]:
        """Generate a random grid while guaranteeing a start-to-goal route.

        A random route is carved first. If the random grid has too many open cells,
        additional open cells outside that route are blocked until the configured cap
        is met. This bounds the exact solver without ever making the task infeasible.
        """
        grid = [["X" if rng.random() < self.config.p_blocked else "O" for _ in range(cols)] for _ in range(rows)]

        start = (rng.randrange(rows), rng.randrange(cols))
        goal = (rng.randrange(rows), rng.randrange(cols))
        while goal == start:
            goal = (rng.randrange(rows), rng.randrange(cols))

        sr, sc = start
        gr, gc = goal
        grid[sr][sc] = "*"
        grid[gr][gc] = "#"

        # Carve a randomized Manhattan path from start to goal. Each step moves
        # closer to the goal, so the carved route is simple and cannot loop.
        guaranteed_path = {start}
        r, c = start
        while (r, c) != goal:
            moves = []
            if r < gr:
                moves.append((r + 1, c))
            elif r > gr:
                moves.append((r - 1, c))
            if c < gc:
                moves.append((r, c + 1))
            elif c > gc:
                moves.append((r, c - 1))
            r, c = rng.choice(moves)
            guaranteed_path.add((r, c))
            if (r, c) != goal:
                grid[r][c] = "O"

        open_cells = [(rr, cc) for rr in range(rows) for cc in range(cols) if grid[rr][cc] != "X"]
        excess = len(open_cells) - self.config.max_open_cells
        if excess > 0:
            removable = [cell for cell in open_cells if cell not in guaranteed_path]
            rng.shuffle(removable)
            # Validation guarantees that the carved route fits under the cap.
            for rr, cc in removable[:excess]:
                grid[rr][cc] = "X"

        return grid

    @staticmethod
    def _get_longest_path_length(matrix: list[list[str]]) -> int:
        """Return the exact maximum number of moves in a simple start-goal path.

        Uses memoized depth-first search over (current cell, visited-cell bitmask).
        Returns -1 only if no path exists; generated tasks always have a path.
        """
        rows, cols = len(matrix), len(matrix[0])
        cells: list[tuple[int, int]] = []
        for r in range(rows):
            for c in range(cols):
                if matrix[r][c] != "X":
                    cells.append((r, c))
        cell_to_id = {cell: i for i, cell in enumerate(cells)}

        start = next(((r, c) for r in range(rows) for c in range(cols) if matrix[r][c] == "*"), None)
        goal = next(((r, c) for r in range(rows) for c in range(cols) if matrix[r][c] == "#"), None)
        if start is None or goal is None:
            return -1

        neighbors: list[list[int]] = [[] for _ in cells]
        for cell, node_id in cell_to_id.items():
            r, c = cell
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (nr, nc) in cell_to_id:
                    neighbors[node_id].append(cell_to_id[(nr, nc)])

        start_id = cell_to_id[start]
        goal_id = cell_to_id[goal]

        @lru_cache(maxsize=None)
        def search(node: int, visited_mask: int) -> int:
            if node == goal_id:
                return 0

            best = -1
            for nxt in neighbors[node]:
                bit = 1 << nxt
                if visited_mask & bit:
                    continue
                remaining = search(nxt, visited_mask | bit)
                if remaining >= 0:
                    best = max(best, remaining + 1)
            return best

        return search(start_id, 1 << start_id)

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """Require the exact answer, allowing surrounding whitespace."""
        if not isinstance(answer, str):
            return 0.0
        return float(answer.strip() == entry["answer"])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Generate a single grid and compute its ground-truth longest path length."""
        rng = Random(self.seed + idx)
        rows = rng.randint(self.config.min_rows, self.config.max_rows)
        cols = rng.randint(self.config.min_cols, self.config.max_cols)
        matrix = self._get_grid(rng, rows, cols)

        longest_length = self._get_longest_path_length(matrix)
        if longest_length < 0:  # Defensive assertion: generation should always carve a route.
            raise RuntimeError("Generated grid unexpectedly has no start-to-goal path")

        open_cell_count = sum(cell != "X" for row in matrix for cell in row)
        return {
            "question": QUESTION_TEMPLATE.format(grid=self._matrix_to_str(matrix)),
            "answer": str(longest_length),
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "matrix": matrix,
                "longest_path_length": longest_length,
                "rows": rows,
                "cols": cols,
                "open_cell_count": open_cell_count,
                "difficulty": {
                    "rows": (self.config.min_rows, self.config.max_rows),
                    "cols": (self.config.min_cols, self.config.max_cols),
                    "p_blocked": self.config.p_blocked,
                    "max_open_cells": self.config.max_open_cells,
                },
            },
        }


register_dataset(DATASET_NAME, LongestPathDataset, LongestPathConfig)
