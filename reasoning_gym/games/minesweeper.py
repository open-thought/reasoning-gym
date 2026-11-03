"""Minesweeper deduction: identify which unrevealed cells are mines."""

import re
from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from ..factory import ProceduralDataset, register_dataset

DATASET_NAME = "minesweeper"


@dataclass
class MinesweeperConfig:
    """Configuration for Minesweeper puzzle generation"""

    min_rows: int = 5
    max_rows: int = 8
    min_cols: int = 5
    max_cols: int = 8
    mine_density: float = 0.2  # fraction of cells that are mines
    hide_safe_prob: float = 0.15  # probability that a safe cell is also hidden
    max_attempts: int = 500  # tries to find a uniquely solvable board
    size: int = 500
    seed: Optional[int] = None

    def validate(self):
        assert 2 <= self.min_rows <= self.max_rows, "need 2 <= min_rows <= max_rows"
        assert 2 <= self.min_cols <= self.max_cols, "need 2 <= min_cols <= max_cols"
        assert 0.0 < self.mine_density < 0.5, "mine_density must be in (0, 0.5)"
        assert 0.0 <= self.hide_safe_prob < 1.0, "hide_safe_prob must be in [0, 1)"
        assert self.max_attempts >= 1, "max_attempts must be >= 1"


def neighbors(r: int, c: int, rows: int, cols: int) -> list[tuple[int, int]]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                out.append((nr, nc))
    return out


def count_solutions(
    rows: int,
    cols: int,
    clues: dict[tuple[int, int], int],
    hidden: list[tuple[int, int]],
    total_mines: int,
    limit: int = 2,
) -> int:
    """Count mine assignments over hidden cells consistent with the clues (stops at `limit`)."""
    hidden_set = set(hidden)
    clue_of = {h: [n for n in neighbors(*h, rows, cols) if n in clues] for h in hidden}
    clue_hidden = {c: [n for n in neighbors(*c, rows, cols) if n in hidden_set] for c in clues}
    assign: dict[tuple[int, int], int] = {}
    found = 0

    def consistent(clue: tuple[int, int]) -> bool:
        hn = clue_hidden[clue]
        mines = sum(1 for h in hn if assign.get(h) == 1)
        unknown = sum(1 for h in hn if h not in assign)
        return mines <= clues[clue] <= mines + unknown

    def backtrack(i: int, mines_so_far: int) -> None:
        nonlocal found
        if found >= limit or mines_so_far > total_mines:
            return
        if i == len(hidden):
            if mines_so_far == total_mines:
                found += 1
            return
        cell = hidden[i]
        for v in (0, 1):
            assign[cell] = v
            if all(consistent(c) for c in clue_of[cell]):
                backtrack(i + 1, mines_so_far + v)
            del assign[cell]

    backtrack(0, 0)
    return found


class MinesweeperDataset(ProceduralDataset):
    """Generates Minesweeper deduction puzzles with a unique solution."""

    def __init__(self, config: MinesweeperConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _make_puzzle(self, rng: Random) -> dict[str, Any]:
        cfg = self.config
        for _ in range(cfg.max_attempts):
            rows = rng.randint(cfg.min_rows, cfg.max_rows)
            cols = rng.randint(cfg.min_cols, cfg.max_cols)
            all_cells = [(r, c) for r in range(rows) for c in range(cols)]
            n_mines = max(1, round(rows * cols * cfg.mine_density))
            mines = set(rng.sample(all_cells, n_mines))
            hidden = set(mines)
            for cell in all_cells:
                if cell not in mines and rng.random() < cfg.hide_safe_prob:
                    hidden.add(cell)

            clues = {
                cell: sum(1 for n in neighbors(*cell, rows, cols) if n in mines)
                for cell in all_cells
                if cell not in hidden
            }
            hidden_list = sorted(hidden)
            if count_solutions(rows, cols, clues, hidden_list, n_mines) == 1:
                board = [["?" if (r, c) in hidden else str(clues[(r, c)]) for c in range(cols)] for r in range(rows)]
                return {"rows": rows, "cols": cols, "board": board, "mines": sorted(mines)}
        raise RuntimeError("Could not generate a uniquely solvable Minesweeper board; relax the config")

    def __getitem__(self, idx: int) -> dict:
        rng = Random(self.seed + idx)
        p = self._make_puzzle(rng)
        board_str = "\n".join(" ".join(row) for row in p["board"])
        n_mines = len(p["mines"])
        answer = ", ".join(f"({r + 1},{c + 1})" for r, c in p["mines"])

        question = (
            f"You are given a Minesweeper board with {p['rows']} rows and {p['cols']} columns.\n"
            "Each cell is either:\n"
            "- a digit: a revealed safe cell showing how many of its (up to 8) neighboring cells are mines, or\n"
            "- '?': an unrevealed cell, which may or may not be a mine.\n\n"
            f"There are exactly {n_mines} mines on the board, and all of them are among the '?' cells.\n\n"
            f"{board_str}\n\n"
            "Rows and columns are numbered starting from 1 (top-left cell is (1,1)).\n"
            "Determine which '?' cells are mines. Give your answer as a list of (row,col) pairs "
            "sorted by row, then column, for example: (1,2), (3,4)"
        )
        return {
            "question": question,
            "answer": answer,
            "metadata": {
                "rows": p["rows"],
                "cols": p["cols"],
                "board": p["board"],
                "mines": [list(m) for m in p["mines"]],  # 0-indexed
                "num_mines": n_mines,
            },
        }

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """1.0 if the set of mine coordinates matches exactly (order/spacing ignored), else 0.0."""
        if not isinstance(answer, str):
            return 0.0
        pattern = r"\(\s*(\d+)\s*,\s*(\d+)\s*\)"
        got = {(int(r) - 1, int(c) - 1) for r, c in re.findall(pattern, answer)}
        expected = {tuple(m) for m in entry["metadata"]["mines"]}
        return 1.0 if got == expected else 0.0


register_dataset(DATASET_NAME, MinesweeperDataset, MinesweeperConfig)
