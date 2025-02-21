"""N Queens puzzle generator

A generalization of the 8-queens puzzle to any board size.
https://en.wikipedia.org/wiki/Eight_queens_puzzle
"""

from copy import deepcopy
from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from ..factory import ProceduralDataset, register_dataset

MIN_BOARD_SIZE = 4
MAX_BOARD_SIZE = 12

QUESTION_TEMPLATE = """Your job is to complete an n x n chess board with n Queens in total, such that no two attack each other.

No two queens attack each other if they are not in the same row, column, or diagonal.

You can place a queen by replacing an underscore (_) with a Q.

Example:
- Input: Given the below board of size 4 x 4 your job is to place 2 queen(s) on the board such that no two queens attack each other.
_ Q _ _
_ _ _ _
_ _ _ _
_ _ Q _
- Output:
_ Q _ _
_ _ _ Q
Q _ _ _
_ _ Q _
- Explanation
    - None of the queens attack each other vertically, horizontally, or diagonally.
    - The added queens are marked with Q at the positions (1, 3) and (2, 0).

Given the below board of size {n} x {n} your job is to place {num_removed} queen(s) on the board such that no two queens attack each other.
{puzzle}
"""


@dataclass
class NQueensConfig:
    """Configuration for N Queens puzzle generation"""

    n: int = 8  # Board size
    min_remove: int = 1  # Minimum number of queens to remove from solved board
    max_remove: int = 7  # Maximum number of queens to remove from solved board

    size: int = 500  # Virtual dataset size
    seed: Optional[int] = None

    def validate(self):
        """Validate configuration parameters"""
        assert MIN_BOARD_SIZE <= self.n <= MAX_BOARD_SIZE, f"n must be between {MIN_BOARD_SIZE} and {MAX_BOARD_SIZE}"
        assert 1 <= self.min_remove <= self.max_remove, "min_remove must be between 1 and max_remove"
        assert self.min_remove <= self.max_remove <= self.n, "max_remove must be between min_remove and n"


class NQueensDataset(ProceduralDataset):
    """Generates N Queens puzzles with configurable difficulty"""

    def __init__(self, config: NQueensConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)
        self._solutions = self._get_all_solutions(config.n)

    def _get_all_solutions(self, n: int) -> list[list[list[str]]]:
        """Get all solutions for the N Queens puzzle"""

        visited_cols = set()
        visited_pos_diag = set()
        visited_neg_diag = set()

        res = []
        board = [["_"] * n for _ in range(n)]

        def backtrack(row: int):
            if row == n:
                res.append(deepcopy(board))
                return

            for col in range(n):
                if col in visited_cols or (row + col) in visited_pos_diag or (row - col) in visited_neg_diag:
                    continue

                visited_cols.add(col)
                visited_pos_diag.add(row + col)
                visited_neg_diag.add(row - col)
                board[row][col] = "Q"
                backtrack(row + 1)
                visited_cols.remove(col)
                visited_pos_diag.remove(row + col)
                visited_neg_diag.remove(row - col)
                board[row][col] = "_"

        backtrack(0)
        return res

    def _create_puzzle(self, solved_board: list[list[str]], num_removed: int, rng: Random) -> list[list[str]]:
        """Create puzzle by removing queens from solved board"""
        puzzle = deepcopy(solved_board)
        queens = [(i, j) for i in range(len(puzzle)) for j in range(len(puzzle)) if puzzle[i][j] == "Q"]
        rng.shuffle(queens)
        for i in range(num_removed):
            x, y = queens[i]
            puzzle[x][y] = "_"
        return puzzle

    def _board_to_string(self, board: list[list[str]]) -> str:
        """Convert board to string representation"""
        return "\n".join(" ".join(x for x in row) for row in board)

    def _string_to_board(self, board_str: str) -> list[list[str]]:
        """Convert string representation to board"""
        return [list(row.split()) for row in board_str.strip().split("\n")]

    def _is_tractable_solution(self, puzzle: list[list[str]], solution: list[list[str]]) -> bool:
        """Check if a solution is achievable from the starting state of the puzzle"""
        for r in range(len(puzzle)):
            for c in range(len(puzzle)):
                if puzzle[r][c] == "Q" and solution[r][c] != "Q":
                    return False
        return True

    def __getitem__(self, idx: int) -> dict:
        """Generate a single N Queens puzzle"""
        rng = Random(self.seed + idx)

        # Randomly select a valid solution
        solved_board = rng.choice(self._solutions)

        # Create puzzle by removing queens
        num_removed = rng.randint(self.config.min_remove, self.config.max_remove)
        puzzle = self._create_puzzle(solved_board, num_removed, rng)
        puzzle_str = self._board_to_string(puzzle)

        # Filter all solutions that are intractable from the puzzle's starting state
        valid_solutions = [board for board in self._solutions if self._is_tractable_solution(puzzle, board)]
        valid_solutions_str = sorted({self._board_to_string(board) for board in valid_solutions})

        return {
            "question": QUESTION_TEMPLATE.format(puzzle=puzzle_str, n=len(puzzle), num_removed=num_removed),
            "answer": rng.choice(valid_solutions_str),  # choose arbitary answer (e.g. for SFT)
            "metadata": {
                "puzzle": puzzle,
                "solutions": valid_solutions,
                "num_removed": num_removed,
                "valid_answers": valid_solutions_str,
            },
        }

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        valid_solutions = entry["metadata"]["valid_answers"]
        if answer is not None:
            if answer in valid_solutions:
                return 1.0
            try:
                answer = self._board_to_string(eval(answer))
                if answer in valid_solutions:
                    return 0.5
            except Exception as e:
                return 0.01
        return 0.0


register_dataset("n_queens", NQueensDataset, NQueensConfig)
