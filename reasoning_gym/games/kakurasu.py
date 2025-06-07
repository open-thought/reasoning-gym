"""
Kakurasu puzzle dataset, adapted for Reasoning Gym from the SynLogic repository: https://github.com/MiniMax-AI/SynLogic/tree/main/games/tasks/kukurasu
"""

from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from ..coaching import BaseCurriculum, RangeAttributeDefinition, ScalarAttributeDefinition
from ..factory import ProceduralDataset, register_dataset

DATASET_NAME = "kakurasu"

PROMPT_TEMPLATES = [
    "You are given a {n_rows} x {n_cols} grid representing a Kukurasu puzzle. In this puzzle, you need to place 1s in the grid so that the weighted sum of each row and column matches the given constraints. The row sums are {row_sums} and the column sums are {col_sums}.\n1. Rules:\n  1. Each cell can contain either a 1 or an 0.\n  2. The weight of a 1 in a row is its column position (1 to {n_cols}).\n  3. The weight of a 1 in a column is its row position (1 to {n_rows}).\n  4. The weighted sum of each row must match the corresponding row constraint.\n  5. The weighted sum of each column must match the corresponding column constraint.\n2. Input:\n{puzzle}",
    "This is a {n_rows} x {n_cols} Kukurasu puzzle grid. Your task is to fill in the grid with 1s and 0s such that the weighted sums match the given constraints. The row sums are {row_sums} and the column sums are {col_sums}.\n1. Rules:\n  1. Each cell must contain either a 1 or an 0.\n  2. In each row, a 1 in position j contributes j points to that row's sum (positions are 1-indexed).\n  3. In each column, a 1 in position i contributes i points to that column's sum (positions are 1-indexed).\n  4. The weighted sum of each row must equal its constraint value.\n  5. The weighted sum of each column must equal its constraint value.\n2. Input:\n{puzzle}",
    "You're presented with a {n_rows} x {n_cols} Kukurasu puzzle grid. The goal is to place 1s in the grid so that the weighted sums of rows and columns match the given constraints: row sums {row_sums} and column sums {col_sums}.\n1. Rules:\n  1. Each cell must be filled with either a 1 or an 0.\n  2. A 1 in column j of any row contributes j points to that row's sum (j ranges from 1 to {n_cols}).\n  3. A 1 in row i of any column contributes i points to that column's sum (i ranges from 1 to {n_rows}).\n  4. Each row's weighted sum must match its constraint value.\n  5. Each column's weighted sum must match its constraint value.\n2. Input:\n{puzzle}",
    "Below is a {n_rows} x {n_cols} Kukurasu puzzle grid. Your objective is to place 1s in the grid such that the weighted sums of rows and columns match the given constraints. Row sums: {row_sums}. Column sums: {col_sums}.\n1. Rules:\n  1. Each cell must contain either a 1 or an 0.\n  2. The weight of a 1 in a row equals its column number (1 to {n_cols}).\n  3. The weight of a 1 in a column equals its row number (1 to {n_rows}).\n  4. The sum of weighted 1s in each row must equal the row constraint.\n  5. The sum of weighted 1s in each column must equal the column constraint.\n2. Input:\n{puzzle}",
    "Here's a {n_rows} x {n_cols} Kukurasu logic puzzle. You need to place 1s in the grid so that the weighted sums match the constraints. Row sums: {row_sums}. Column sums: {col_sums}.\n1. Rules:\n  1. Each cell can be filled with either a 1 or an 0.\n  2. A 1 in the jth position of a row contributes j points to that row's sum.\n  3. A 1 in the ith position of a column contributes i points to that column's sum.\n  4. The weighted sum of each row must equal its constraint value.\n  5. The weighted sum of each column must equal its constraint value.\n2. Input:\n{puzzle}",
    "I'm presenting you with a {n_rows} x {n_cols} Kukurasu puzzle. Your task is to place 1s in the grid so that the weighted sums match the given constraints: row sums {row_sums} and column sums {col_sums}.\n1. Rules:\n  1. Each cell must be filled with either a 1 or an 0.\n  2. In each row, a 1 in position j has a weight of j (where j ranges from 1 to {n_cols}).\n  3. In each column, a 1 in position i has a weight of i (where i ranges from 1 to {n_rows}).\n  4. The weighted sum of each row must match its constraint.\n  5. The weighted sum of each column must match its constraint.\n2. Input:\n{puzzle}",
    "Consider this {n_rows} x {n_cols} Kukurasu puzzle grid. You need to place 1s in the grid such that the weighted sums match the constraints. Row sums: {row_sums}. Column sums: {col_sums}.\n1. Rules:\n  1. Each cell must contain either a 1 or an 0.\n  2. A 1 in column position j contributes j points to its row's sum.\n  3. A 1 in row position i contributes i points to its column's sum.\n  4. Each row's weighted sum must equal its constraint value.\n  5. Each column's weighted sum must equal its constraint value.\n2. Input:\n{puzzle}",
    "You have a {n_rows} x {n_cols} Kukurasu puzzle grid. Your goal is to place 1s in the grid so that the weighted sums match the given constraints: row sums {row_sums} and column sums {col_sums}.\n1. Rules:\n  1. Each cell must be filled with either a 1 or an 0.\n  2. The weight of a 1 in a row is its column position (1 to {n_cols}).\n  3. The weight of a 1 in a column is its row position (1 to {n_rows}).\n  4. The weighted sum of each row must match its constraint.\n  5. The weighted sum of each column must match its constraint.\n2. Input:\n{puzzle}",
    "This {n_rows} x {n_cols} grid represents a Kukurasu puzzle. Your task is to place 1s in the grid so that the weighted sums match the constraints. Row sums: {row_sums}. Column sums: {col_sums}.\n1. Rules:\n  1. Each cell must contain either a 1 or an 0.\n  2. A 1 in the jth position of a row contributes j points to that row's sum.\n  3. A 1 in the ith position of a column contributes i points to that column's sum.\n  4. The weighted sum of each row must equal its constraint value.\n  5. The weighted sum of each column must equal its constraint value.\n2. Input:\n{puzzle}",
    "Examine this {n_rows} x {n_cols} Kukurasu puzzle grid. Your objective is to place 1s in the grid such that the weighted sums match the given constraints: row sums {row_sums} and column sums {col_sums}.\n1. Rules:\n  1. Each cell must be filled with either a 1 or an 0.\n  2. The weight of a 1 in a row equals its column number (1 to {n_cols}).\n  3. The weight of a 1 in a column equals its row number (1 to {n_rows}).\n  4. The weighted sum of each row must match its constraint.\n  5. The weighted sum of each column must match its constraint.\n2. Input:\n{puzzle}",
]


@dataclass
class KakurasuConfig:
    min_rows: int = 4
    max_rows: int = 5
    min_cols: int = 4
    max_cols: int = 5
    p_ones: float = 0.3
    seed: Optional[int] = None
    size: int = 500  # Virtual dataset size
    max_retries: int = 1000  # Max retries to find a unique puzzle. If exceeded, a non-unique puzzle may be returned

    def validate(self):
        """Validate configuration parameters"""
        assert 3 <= self.min_rows <= 9, "n_rows must be between 3 and 9"
        assert 3 <= self.max_rows <= 9, "n_cols must be between 3 and 9"
        assert 3 <= self.min_rows <= 9, "n_rows must be between 3 and 9"
        assert 3 <= self.max_cols <= 9, "n_cols must be between 3 and 9"
        assert self.min_rows <= self.max_rows, "min_rows must be less than or equal to max_rows"
        assert self.min_cols <= self.max_cols, "min_cols must be less than or equal to max_cols"
        assert 0 <= self.p_ones <= 1, "p_ones must be between 0 and 1"


class KakurasuDataset(ProceduralDataset):
    """Generates Kakurasu puzzles with configurable size."""

    def __init__(self, config: KakurasuConfig):
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
        """Generate Kakurasu puzzles that have at least one solution."""
        rng = Random(self.seed + idx)

        n_rows = rng.randint(self.config.min_rows, self.config.max_rows)
        n_cols = rng.randint(self.config.min_cols, self.config.max_cols)

        for retry in range(self.config.max_retries):
            solution_grid = self._generate_random_grid(rng, n_rows, n_cols)
            self._repair_grid(rng, solution_grid)

            row_sums, col_sums = self._calculate_row_col_sums(solution_grid)
            empty_grid = [[0 for _ in range(n_cols)] for _ in range(n_rows)]

            if retry < self.config.max_retries - 1:
                if 0 in row_sums or 0 in col_sums or sum(row_sums) != sum(col_sums):
                    continue
                if self._count_solutions(n_rows, n_cols, row_sums, col_sums) != 1:
                    continue

            prompt = rng.choice(PROMPT_TEMPLATES).format(
                n_rows=n_rows,
                n_cols=n_cols,
                row_sums=row_sums,
                col_sums=col_sums,
                puzzle="\n".join(
                    [" ".join(str(cell) for cell in row) for row in empty_grid],
                ),
            )

            return {
                "question": prompt,
                "answer": "\n".join(
                    [" ".join(str(cell) for cell in row) for row in solution_grid],
                ),
                "metadata": {
                    "source_dataset": DATASET_NAME,
                    "source_idx": idx,
                    "n_rows": n_rows,
                    "n_cols": n_cols,
                    "p_ones": self.config.p_ones,
                    "puzzle": empty_grid,
                    "row_sums": row_sums,
                    "col_sums": col_sums,
                    "solution": solution_grid,
                    "difficulty": {
                        "rows": (self.config.min_rows, self.config.max_rows),
                        "cols": (self.config.min_cols, self.config.max_cols),
                        "p_ones": self.config.p_ones,
                    },
                },
            }

    def _generate_random_grid(self, rng: Random, n_rows: int, n_cols: int) -> list[list[int]]:
        """Generate a random valid solution grid."""
        return [[1 if rng.random() < self.config.p_ones else 0 for _ in range(n_cols)] for _ in range(n_rows)]

    def _calculate_row_col_sums(self, grid) -> tuple[list[int], list[int]]:
        """Calculate row and column sums based on the solution grid"""
        n_rows = len(grid)
        n_cols = len(grid[0]) if n_rows > 0 else 0
        row_sums = [sum((j + 1) for j, cell in enumerate(row) if cell == 1) for row in grid]
        col_sums = [sum((i + 1) for i in range(n_rows) if grid[i][j] == 1) for j in range(n_cols)]
        return row_sums, col_sums

    def _repair_grid(self, rng: Random, grid: list[list[int]]):
        """Ensure every row/col has at least one '1'."""
        n_rows = len(grid)
        n_cols = len(grid[0]) if n_rows > 0 else 0

        for i, row in enumerate(grid):
            if 1 not in row:
                grid[i][rng.randrange(n_cols)] = 1

        for j in range(n_cols):
            if all(grid[i][j] == 0 for i in range(n_rows)):
                grid[rng.randrange(n_rows)][j] = 1

    def _count_solutions(self, n: int, m: int, row_sums: list[int], col_sums: list[int], limit: int = 2) -> int:
        """Return number of solutions, stopping at `limit`."""
        row_patterns: list[list[list[int]]] = []
        for target in row_sums:
            patterns = []
            for mask in range(1 << m):
                if sum(((j + 1) if (mask >> j) & 1 else 0) for j in range(m)) == target:
                    patterns.append([(mask >> j) & 1 for j in range(m)])
            if not patterns:
                return 0
            row_patterns.append(patterns)

        col_remaining = col_sums[:]
        solutions = 0

        def dfs(r: int):
            nonlocal solutions
            if solutions >= limit:
                return
            if r == n:
                if all(c == 0 for c in col_remaining):
                    solutions += 1
                return
            for pat in row_patterns[r]:
                ok = True
                for j, bit in enumerate(pat):
                    col_remaining[j] -= (r + 1) * bit
                    if col_remaining[j] < 0:
                        ok = False
                if ok:
                    dfs(r + 1)
                for j, bit in enumerate(pat):
                    col_remaining[j] += (r + 1) * bit
                if solutions >= limit:
                    break

        dfs(0)
        return solutions

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        if not isinstance(answer, str):
            return 0.0

        metadata = entry["metadata"]
        row_sums, col_sums = metadata["row_sums"], metadata["col_sums"]
        n_rows, n_cols = metadata["n_rows"], metadata["n_cols"]

        try:
            grid = self._parse_grid(answer)

            if len(grid) != n_rows or any(len(row) != n_cols for row in grid):
                return 0.0

            if any(cell not in [1, 0] for row in grid for cell in row):
                return 0.0

            ans_row_sums = [sum((j + 1) for j, cell in enumerate(row) if cell == 1) for row in grid]

            if ans_row_sums != row_sums:
                return 0.0

            ans_col_sums = [sum((i + 1) for i in range(n_rows) if grid[i][j] == 1) for j in range(n_cols)]

            if ans_col_sums != col_sums:
                return 0.0

            return 1.0
        except Exception:
            return 0.0

    def _parse_grid(self, answer: str) -> list[list[str]]:
        grid = []
        for line in answer.strip().split("\n"):
            grid.append([int(c) for c in line.strip() if c in "01"])
        return grid


class KakurasuCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(KakurasuCurriculum.__name__, KakurasuConfig)

        self._define_attributes(
            RangeAttributeDefinition(
                name="rows",
                levels=[4, 6, 7, 9],
                description="Row count",
                lower_field_name="min_rows",
                upper_field_name="max_rows",
            ),
            RangeAttributeDefinition(
                name="cols",
                levels=[4, 6, 7, 9],
                description="Column count",
                lower_field_name="min_cols",
                upper_field_name="max_cols",
            ),
            ScalarAttributeDefinition(
                name="p_ones",
                levels=[0.50, 0.40, 0.30, 0.20],
                description="Probability of a cell being filled",
                field_name="p_ones",
            ),
        )


register_dataset(DATASET_NAME, KakurasuDataset, KakurasuConfig, KakurasuCurriculum)
