"""Mini Sudoku (4x4) puzzle generator"""

from dataclasses import dataclass
from random import Random
from typing import List, Optional, Tuple

from ..factory import ProceduralDataset, register_dataset


@dataclass
class MiniSudokuConfig:
    """Configuration for 4x4 sudoku puzzle generation"""

    min_empty: int = 8  # Minimum number of empty cells
    max_empty: int = 12  # Maximum number of empty cells
    seed: Optional[int] = None
    size: int = 500  # Virtual dataset size

    def validate(self):
        """Validate configuration parameters"""
        assert 0 <= self.min_empty <= 16, "min_empty must be between 0 and 16"
        assert self.min_empty <= self.max_empty <= 16, "max_empty must be between min_empty and 16"


class MiniSudokuDataset(ProceduralDataset):
    """Generates 4x4 sudoku puzzles with configurable difficulty"""

    def __init__(self, config: MiniSudokuConfig):
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
        if num in [board[i][col] for i in range(4)]:
            return False

        # Check 2x2 box
        box_row, box_col = 2 * (row // 2), 2 * (col // 2)
        for i in range(box_row, box_row + 2):
            for j in range(box_col, box_col + 2):
                if board[i][j] == num:
                    return False
        return True

    def _solve(self, board: List[List[int]]) -> bool:
        """Solve mini sudoku using backtracking"""
        empty = self._find_empty(board)
        if not empty:
            return True

        row, col = empty
        for num in range(1, 5):
            if self._is_valid(board, row, col, num):
                board[row][col] = num
                if self._solve(board):
                    return True
                board[row][col] = 0
        return False

    def _find_empty(self, board: List[List[int]]) -> Optional[Tuple[int, int]]:
        """Find an empty cell"""
        for i in range(4):
            for j in range(4):
                if board[i][j] == 0:
                    return (i, j)
        return None

    def _generate_solved_board(self, rng: Random) -> List[List[int]]:
        """Generate a complete solved mini sudoku board"""
        board = [[0] * 4 for _ in range(4)]

        # Try multiple times to generate a valid board
        max_attempts = 100
        for _ in range(max_attempts):
            # Start fresh
            for i in range(4):
                for j in range(4):
                    board[i][j] = 0

            # Fill diagonal boxes first (they are independent)
            for i in range(0, 4, 2):
                nums = list(range(1, 5))
                rng.shuffle(nums)
                pos = 0
                for r in range(i, i + 2):
                    for c in range(i, i + 2):
                        board[r][c] = nums[pos]
                        pos += 1

            # Try to solve the rest
            if self._solve(board):
                return board

        raise RuntimeError("Failed to generate valid mini sudoku board")

    def _create_puzzle(self, solved_board: List[List[int]], num_empty: int, rng: Random) -> List[List[int]]:
        """Create puzzle by removing numbers from solved board"""
        puzzle = [row[:] for row in solved_board]
        cells = [(i, j) for i in range(4) for j in range(4)]
        rng.shuffle(cells)

        for i, j in cells[:num_empty]:
            puzzle[i][j] = 0

        return puzzle

    def _board_to_string(self, board: List[List[int]]) -> str:
        """Convert board to string representation"""
        return "\n".join(" ".join(str(x) if x != 0 else "_" for x in row) for row in board)

    def __getitem__(self, idx: int) -> dict:
        """Generate a single mini sudoku puzzle"""
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
            "question": f"Solve this 4x4 Mini Sudoku puzzle:\n{puzzle_str}",
            "answer": solution_str,
            "metadata": {"puzzle": puzzle, "solution": solved_board, "num_empty": num_empty},
        }


register_dataset("mini_sudoku", MiniSudokuDataset, MiniSudokuConfig)
