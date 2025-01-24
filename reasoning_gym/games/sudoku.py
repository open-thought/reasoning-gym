"""Sudoku puzzle generator"""

from dataclasses import dataclass
from random import Random
from typing import List, Optional, Tuple

from ..factory import ProceduralDataset, register_dataset


@dataclass
class SudokuConfig:
    """Configuration for sudoku puzzle generation"""

    min_empty: int = 30  # Minimum number of empty cells
    max_empty: int = 50  # Maximum number of empty cells
    seed: Optional[int] = None
    size: int = 500  # Virtual dataset size

    def validate(self):
        """Validate configuration parameters"""
        assert 0 <= self.min_empty <= 81, "min_empty must be between 0 and 81"
        assert self.min_empty <= self.max_empty <= 81, "max_empty must be between min_empty and 81"


class SudokuDataset(ProceduralDataset):
    """Generates sudoku puzzles with configurable difficulty"""

    def __init__(self, config: SudokuConfig):
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

    def _is_valid(self, board: List[List[int]], row: int, col: int, num: int) -> bool:
        """Check if number can be placed at position"""
        # Check row
        if num in board[row]:
            return False

        # Check column
        if num in [board[i][col] for i in range(9)]:
            return False

        # Check 3x3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(box_row, box_row + 3):
            for j in range(box_col, box_col + 3):
                if board[i][j] == num:
                    return False
        return True

    def _solve(self, board: List[List[int]]) -> bool:
        """Solve sudoku using backtracking"""
        empty = self._find_empty(board)
        if not empty:
            return True

        row, col = empty
        for num in range(1, 10):
            if self._is_valid(board, row, col, num):
                board[row][col] = num
                if self._solve(board):
                    return True
                board[row][col] = 0
        return False

    def _find_empty(self, board: List[List[int]]) -> Optional[Tuple[int, int]]:
        """Find an empty cell"""
        for i in range(9):
            for j in range(9):
                if board[i][j] == 0:
                    return (i, j)
        return None

    def _generate_solved_board(self, rng: Random) -> List[List[int]]:
        """Generate a complete solved sudoku board"""
        board = [[0] * 9 for _ in range(9)]

        # Fill diagonal boxes first (they are independent)
        for i in range(0, 9, 3):
            nums = list(range(1, 10))
            rng.shuffle(nums)
            pos = 0
            for r in range(i, i + 3):
                for c in range(i, i + 3):
                    board[r][c] = nums[pos]
                    pos += 1

        # Solve the rest
        self._solve(board)
        return board

    def _create_puzzle(self, solved_board: List[List[int]], num_empty: int, rng: Random) -> List[List[int]]:
        """Create puzzle by removing numbers from solved board"""
        puzzle = [row[:] for row in solved_board]
        cells = [(i, j) for i in range(9) for j in range(9)]
        rng.shuffle(cells)

        for i, j in cells[:num_empty]:
            puzzle[i][j] = 0

        return puzzle

    def _board_to_string(self, board: List[List[int]]) -> str:
        """Convert board to string representation"""
        return "\n".join(" ".join(str(x) if x != 0 else "_" for x in row) for row in board)

    def __getitem__(self, idx: int) -> dict:
        """Generate a single sudoku puzzle"""
        rng = Random(self.seed + idx)

        # Generate solved board
        solved_board = self._generate_solved_board(rng)

        # Create puzzle by removing numbers
        num_empty = rng.randint(self.config.min_empty, self.config.max_empty)
        puzzle = self._create_puzzle(solved_board, num_empty, rng)

        # Format as strings
        puzzle_str = self._board_to_string(puzzle)
        solution_str = self._board_to_string(solved_board)

        return {
            "question": f"Solve this Sudoku puzzle:\n{puzzle_str}",
            "answer": solution_str,
            "metadata": {"puzzle": puzzle, "solution": solved_board, "num_empty": num_empty},
        }


register_dataset("sudoku", SudokuDataset, SudokuConfig)
