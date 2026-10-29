"""
Nonogram (a.k.a. Picross / Griddlers / Paint-by-Numbers) puzzle dataset.

A hidden black-and-white picture is described only by run-length clues: for every row and
every column, the clue lists the lengths of the consecutive blocks of filled cells in order.
The solver must reconstruct a grid that satisfies all row and column clues simultaneously.

The task is verified by checking the submitted grid against the clues, so *any* grid that
satisfies every clue receives full reward, even if the generator's hidden picture differs.
Optionally (default), the generator only emits puzzles with a unique solution.
"""

from dataclasses import dataclass
from functools import lru_cache
from random import Random
from typing import Any, Optional

from ..coaching import BaseCurriculum, RangeAttributeDefinition, ScalarAttributeDefinition
from ..factory import ProceduralDataset, register_dataset

DATASET_NAME = "nonogram"

FILLED_CHARS = frozenset("1#Xx■█")
EMPTY_CHARS = frozenset("0._-□")

RULES = (
    "Rules:\n"
    "1. Every cell is either filled or empty.\n"
    "2. The clue for a row lists, from left to right, the lengths of the runs of consecutive filled cells in that row.\n"
    "3. The clue for a column lists, from top to bottom, the lengths of the runs of consecutive filled cells in that column.\n"
    "4. Runs in the same line are separated by at least one empty cell.\n"
    "5. A clue of 0 means the line contains no filled cells."
)

ANSWER_FORMAT = (
    "Give your answer as the completed grid: exactly {n_rows} lines (top row first), each containing {n_cols} "
    "values separated by single spaces, using 1 for a filled cell and 0 for an empty cell. "
    "Do not include clues, row labels, or any other text in the grid."
)

PROMPT_TEMPLATES = [
    "Solve the following {n_rows} x {n_cols} Nonogram puzzle.\n\n{rules}\n\nRow clues (top to bottom):\n{row_clues}\n\n"
    "Column clues (left to right):\n{col_clues}\n\n{answer_format}",
    "You are given a Nonogram (also called Picross or Paint-by-Numbers) on a grid with {n_rows} rows and {n_cols} "
    "columns. Fill in cells so that every row and column matches its clue.\n\n{rules}\n\n"
    "Row clues (top to bottom):\n{row_clues}\n\nColumn clues (left to right):\n{col_clues}\n\n{answer_format}",
    "A hidden picture is drawn on a {n_rows}-row by {n_cols}-column grid. Reconstruct it from the run-length clues "
    "below.\n\n{rules}\n\nColumn clues (left to right):\n{col_clues}\n\nRow clues (top to bottom):\n{row_clues}\n\n"
    "{answer_format}",
    "Here is a Nonogram logic puzzle with {n_rows} rows and {n_cols} columns. Determine which cells are filled.\n\n"
    "{rules}\n\nRow clues (top to bottom):\n{row_clues}\n\nColumn clues (left to right):\n{col_clues}\n\n"
    "{answer_format}",
]


def compute_line_clue(line: list[int]) -> list[int]:
    """Return the run-length clue for a single line of 0/1 cells ([] if the line is empty)."""
    clue, run = [], 0
    for cell in line:
        if cell == 1:
            run += 1
        elif run:
            clue.append(run)
            run = 0
    if run:
        clue.append(run)
    return clue


def compute_clues(grid: list[list[int]]) -> tuple[list[list[int]], list[list[int]]]:
    """Return (row_clues, col_clues) for a rectangular 0/1 grid."""
    row_clues = [compute_line_clue(row) for row in grid]
    col_clues = [compute_line_clue([row[j] for row in grid]) for j in range(len(grid[0]))]
    return row_clues, col_clues


def _line_possibilities(line: tuple[int, ...], clue: tuple[int, ...]) -> Optional[tuple[list[bool], list[bool]]]:
    """Exact line solver.

    Given a partially known line (-1 unknown, 0 empty, 1 filled) and its clue, compute for every
    cell whether it can be filled and whether it can be empty in at least one arrangement that is
    consistent with both the clue and the known cells. Returns None if no arrangement exists.
    Runs in O(len(line) * len(clue)) time via memoised search over (position, block index).
    """
    n, k = len(line), len(clue)

    # zeros_prefix[i] = number of known-empty cells in line[:i] (to test block placement in O(1))
    zeros_prefix = [0] * (n + 1)
    for i, cell in enumerate(line):
        zeros_prefix[i + 1] = zeros_prefix[i] + (cell == 0)

    def block_fits(start: int, length: int) -> bool:
        end = start + length
        return end <= n and zeros_prefix[end] == zeros_prefix[start]

    @lru_cache(maxsize=None)
    def feasible(i: int, j: int) -> bool:
        """Can blocks clue[j:] be placed in line[i:]?"""
        if i >= n:
            return j == k
        if j == k:
            return all(line[x] != 1 for x in range(i, n))
        # Option A: cell i is empty
        if line[i] != 1 and feasible(i + 1, j):
            return True
        # Option B: block j starts at cell i
        end = i + clue[j]
        if block_fits(i, clue[j]):
            if end == n:
                return j + 1 == k
            if line[end] != 1 and feasible(end + 1, j + 1):
                return True
        return False

    if not feasible(0, 0):
        return None

    can_fill = [False] * n
    can_empty = [False] * n
    visited: set[tuple[int, int]] = set()
    stack = [(0, 0)]
    while stack:
        i, j = stack.pop()
        if (i, j) in visited or i >= n:
            continue
        visited.add((i, j))
        if j == k:
            for x in range(i, n):
                can_empty[x] = True
            continue
        if line[i] != 1 and feasible(i + 1, j):
            can_empty[i] = True
            stack.append((i + 1, j))
        end = i + clue[j]
        if block_fits(i, clue[j]):
            if end == n and j + 1 == k:
                for x in range(i, end):
                    can_fill[x] = True
            elif end < n and line[end] != 1 and feasible(end + 1, j + 1):
                for x in range(i, end):
                    can_fill[x] = True
                can_empty[end] = True
                stack.append((end + 1, j + 1))
    return can_fill, can_empty


class _SearchBudgetExceeded(Exception):
    pass


def count_solutions(
    row_clues: list[list[int]], col_clues: list[list[int]], limit: int = 2, max_nodes: Optional[int] = 5000
) -> Optional[int]:
    """Count grids satisfying the clues, stopping once `limit` solutions have been found.

    Uses exact line-solving constraint propagation, falling back to branching on an
    undetermined cell when propagation stalls. Branching is exponential in the worst case, so the
    search visits at most `max_nodes` nodes (None = unbounded); if the budget runs out the
    function returns None ("unknown").
    """
    n_rows, n_cols = len(row_clues), len(col_clues)
    row_clues_t = [tuple(c) for c in row_clues]
    col_clues_t = [tuple(c) for c in col_clues]

    def propagate(grid: list[list[int]]) -> bool:
        dirty_rows, dirty_cols = set(range(n_rows)), set(range(n_cols))
        while dirty_rows or dirty_cols:
            if dirty_rows:
                r = dirty_rows.pop()
                res = _line_possibilities(tuple(grid[r]), row_clues_t[r])
                if res is None:
                    return False
                can_fill, can_empty = res
                for c in range(n_cols):
                    if grid[r][c] == -1:
                        if can_fill[c] and not can_empty[c]:
                            grid[r][c] = 1
                            dirty_cols.add(c)
                        elif can_empty[c] and not can_fill[c]:
                            grid[r][c] = 0
                            dirty_cols.add(c)
            else:
                c = dirty_cols.pop()
                res = _line_possibilities(tuple(grid[r][c] for r in range(n_rows)), col_clues_t[c])
                if res is None:
                    return False
                can_fill, can_empty = res
                for r in range(n_rows):
                    if grid[r][c] == -1:
                        if can_fill[r] and not can_empty[r]:
                            grid[r][c] = 1
                            dirty_rows.add(r)
                        elif can_empty[r] and not can_fill[r]:
                            grid[r][c] = 0
                            dirty_rows.add(r)
        return True

    found = 0
    nodes = 0

    def pick_branch_cell(grid: list[list[int]]) -> Optional[tuple[int, int]]:
        # Branch inside the row with the fewest (but > 0) unknown cells: its options are most constrained.
        best, best_unknown = None, n_cols + 1
        for r in range(n_rows):
            unknown = grid[r].count(-1)
            if 0 < unknown < best_unknown:
                best, best_unknown = (r, grid[r].index(-1)), unknown
        return best

    def search(grid: list[list[int]]) -> None:
        nonlocal found, nodes
        nodes += 1
        if max_nodes is not None and nodes > max_nodes:
            raise _SearchBudgetExceeded
        if found >= limit or not propagate(grid):
            return
        cell = pick_branch_cell(grid)
        if cell is None:
            found += 1  # fully determined and consistent
            return
        r, c = cell
        for value in (1, 0):
            child = [row[:] for row in grid]
            child[r][c] = value
            search(child)
            if found >= limit:
                return

    try:
        search([[-1] * n_cols for _ in range(n_rows)])
    except _SearchBudgetExceeded:
        return None
    return found


@dataclass
class NonogramConfig:
    """Configuration for Nonogram puzzle generation."""

    min_rows: int = 5
    max_rows: int = 5
    min_cols: int = 5
    max_cols: int = 5
    fill_prob: float = 0.6  # probability that a cell of the hidden picture is filled
    require_unique: bool = True  # only emit puzzles with exactly one solution (best effort)
    max_retries: int = 100  # attempts to find a unique puzzle before falling back to a non-unique one
    max_search_nodes: int = 300  # branching budget of the uniqueness check per attempt (bounds generation time)
    seed: Optional[int] = None
    size: int = 500

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.size > 0, "size must be positive"
        assert 2 <= self.min_rows <= self.max_rows <= 25, "need 2 <= min_rows <= max_rows <= 25"
        assert 2 <= self.min_cols <= self.max_cols <= 25, "need 2 <= min_cols <= max_cols <= 25"
        assert 0.0 < self.fill_prob < 1.0, "fill_prob must be in (0, 1)"
        assert self.max_retries > 0, "max_retries must be positive"
        assert self.max_search_nodes > 0, "max_search_nodes must be positive"


class NonogramDataset(ProceduralDataset):
    """Generates Nonogram puzzles: reconstruct a binary grid from row and column run-length clues."""

    def __init__(self, config: NonogramConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def __getitem__(self, idx: int) -> dict:
        rng = Random(self.seed + idx)

        n_rows = rng.randint(self.config.min_rows, self.config.max_rows)
        n_cols = rng.randint(self.config.min_cols, self.config.max_cols)

        grid = None
        unique = False
        for _ in range(self.config.max_retries):
            candidate = self._random_grid(rng, n_rows, n_cols)
            filled = sum(map(sum, candidate))
            if filled == 0 or filled == n_rows * n_cols:
                continue  # trivial puzzle
            grid = candidate
            if not self.config.require_unique:
                break
            row_clues, col_clues = compute_clues(candidate)
            if count_solutions(row_clues, col_clues, limit=2, max_nodes=self.config.max_search_nodes) == 1:
                unique = True
                break
        if grid is None:  # extremely unlikely: only trivial grids were drawn
            grid = self._random_grid(rng, n_rows, n_cols)
            grid[rng.randrange(n_rows)][rng.randrange(n_cols)] ^= 1

        row_clues, col_clues = compute_clues(grid)
        if not self.config.require_unique:
            unique = count_solutions(row_clues, col_clues, limit=2, max_nodes=self.config.max_search_nodes) == 1

        question = rng.choice(PROMPT_TEMPLATES).format(
            n_rows=n_rows,
            n_cols=n_cols,
            rules=RULES,
            row_clues=self._format_clues(row_clues, "Row"),
            col_clues=self._format_clues(col_clues, "Column"),
            answer_format=ANSWER_FORMAT.format(n_rows=n_rows, n_cols=n_cols),
        )

        return {
            "question": question,
            "answer": self._format_grid(grid),
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "n_rows": n_rows,
                "n_cols": n_cols,
                "fill_prob": self.config.fill_prob,
                "row_clues": row_clues,
                "col_clues": col_clues,
                "solution": grid,
                "unique_solution": unique,
                "difficulty": {
                    "rows": (self.config.min_rows, self.config.max_rows),
                    "cols": (self.config.min_cols, self.config.max_cols),
                    "fill_prob": self.config.fill_prob,
                },
            },
        }

    def _random_grid(self, rng: Random, n_rows: int, n_cols: int) -> list[list[int]]:
        p = self.config.fill_prob
        return [[1 if rng.random() < p else 0 for _ in range(n_cols)] for _ in range(n_rows)]

    @staticmethod
    def _format_clues(clues: list[list[int]], label: str) -> str:
        return "\n".join(f"{label} {i + 1}: {' '.join(map(str, c)) if c else '0'}" for i, c in enumerate(clues))

    @staticmethod
    def _format_grid(grid: list[list[int]]) -> str:
        return "\n".join(" ".join(str(cell) for cell in row) for row in grid)

    @staticmethod
    def _parse_grid(answer: str) -> list[list[int]]:
        """Parse a grid answer leniently.

        Accepts 1/0, #/., X/., and similar cell symbols, with or without separators. Lines without any
        cell symbol (e.g. code fences) are skipped, and anything up to a trailing ':' (e.g. "Row 3:") is dropped.
        """
        grid = []
        for raw_line in answer.strip().splitlines():
            line = raw_line.rsplit(":", 1)[-1]
            cells = [1 if ch in FILLED_CHARS else 0 for ch in line if ch in FILLED_CHARS or ch in EMPTY_CHARS]
            if cells:
                grid.append(cells)
        return grid

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """Return 1.0 iff the answer is a grid of the right shape that satisfies every row and column clue."""
        if not isinstance(answer, str) or not answer.strip():
            return 0.0
        metadata = entry["metadata"]
        n_rows, n_cols = metadata["n_rows"], metadata["n_cols"]
        try:
            grid = self._parse_grid(answer)
        except Exception:
            return 0.0
        if len(grid) != n_rows or any(len(row) != n_cols for row in grid):
            return 0.0
        row_clues, col_clues = compute_clues(grid)
        if row_clues == metadata["row_clues"] and col_clues == metadata["col_clues"]:
            return 1.0
        return 0.0


class NonogramCurriculum(BaseCurriculum):
    """Curriculum that grows the grid and makes the hidden picture sparser (fewer forced cells)."""

    def __init__(self):
        super().__init__(NonogramCurriculum.__name__, NonogramConfig)

        self._define_attributes(
            RangeAttributeDefinition(
                name="rows",
                levels=[4, 5, 7, 10, 12, 15],
                description="Number of rows in the grid",
                lower_field_name="min_rows",
                upper_field_name="max_rows",
            ),
            RangeAttributeDefinition(
                name="cols",
                levels=[4, 5, 7, 10, 12, 15],
                description="Number of columns in the grid",
                lower_field_name="min_cols",
                upper_field_name="max_cols",
            ),
            ScalarAttributeDefinition(
                name="fill_prob",
                levels=[0.7, 0.6, 0.55, 0.5],
                description="Fill density of the hidden picture (sparser pictures give weaker clues)",
                field_name="fill_prob",
            ),
        )


register_dataset(DATASET_NAME, NonogramDataset, NonogramConfig, NonogramCurriculum)
