"""Mini Sudoku (4x4) puzzle generator"""

import copy
from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from ..factory import ProceduralDataset, register_dataset


@dataclass
class MiniSudokuConfig:
    """Configuration for 4x4 sudoku puzzle generation"""

    min_empty: int = (
        8  # Minimum number of empty cells. Occasionally this can be violated, if removing more cells would break the puzzle's uniqueness.
    )
    max_empty: int = 12  # Maximum number of empty cells
    seed: Optional[int] = None
    size: int = 500  # Virtual dataset size

    def validate(self):
        """Validate configuration parameters"""
        # More than 12 empty cells is incompatible with a unique solution
        assert 0 <= self.min_empty <= 12, "min_empty must be between 0 and 12"
        assert self.min_empty <= self.max_empty <= 12, "max_empty must be between min_empty and 12"


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

    def _is_valid(self, board: list[list[int]], row: int, col: int, num: int) -> bool:
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

    def _solve(self, board: list[list[int]]) -> bool:
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

    def _find_empty(self, board: list[list[int]]) -> Optional[tuple[int, int]]:
        """Find an empty cell"""
        for i in range(4):
            for j in range(4):
                if board[i][j] == 0:
                    return (i, j)
        return None

    def _generate_solved_board(self, rng: Random) -> list[list[int]]:
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

    def _count_solutions(self, board: list[list[int]], limit: int = 2) -> int:
        """Count the number of solutions for a given board"""

        def _count_solutions_helper(board: list[list[int]]) -> int:
            empty = self._find_empty(board)
            if not empty:
                return 1

            row, col = empty
            count = 0
            for num in range(1, 5):
                if self._is_valid(board, row, col, num):
                    board[row][col] = num
                    count += _count_solutions_helper(board)
                    if count >= limit:
                        return count
                    board[row][col] = 0
            return count

        return _count_solutions_helper(board)

    def _create_puzzle(self, solved_board: list[list[int]], num_empty: int, rng: Random) -> list[list[int]]:
        """Create puzzle by removing numbers from solved board"""
        puzzle = [row[:] for row in solved_board]
        cells = [(i, j) for i in range(4) for j in range(4)]
        rng.shuffle(cells)
        num_removed = 0

        for i, j in cells:
            saved = puzzle[i][j]
            puzzle[i][j] = 0
            puzzle_copy = copy.deepcopy(puzzle)
            # Check if removing this clue breaks uniqueness
            if self._count_solutions(puzzle_copy) > 1:
                puzzle[i][j] = saved
            else:
                num_removed += 1
                if num_removed == num_empty:
                    break

        return puzzle

    def _board_to_string(self, board: list[list[int]]) -> str:
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

        # Update the num_empty to be used in the metadata if we couldn't remove as many as we wanted
        num_empty = sum(1 for row in puzzle for x in row if x == 0)

        # Format as strings
        puzzle_str = self._board_to_string(puzzle)
        solution_str = self._board_to_string(solved_board)

        question = (
            "In 4x4 Mini Sudoku:\n"
            "- Each row must contain each number from 1-4 exactly once\n"
            "- Each column must contain each number 1-4 exactly once\n"
            "- Each 2x2 subgrid must contain each number 1-4 exactly once\n"
            f"Solve this 4x4 Mini Sudoku puzzle:\n{puzzle_str}\n"
            "Format your response as the puzzle above, with spaces separating each number within a row, and newlines separating rows.\n"
        )

        return {
            "question": question,
            "answer": solution_str,
            "metadata": {"puzzle": puzzle, "solution": solved_board, "num_empty": num_empty},
        }

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        if not answer:
            return 0.0

        oracle_answer = entry["answer"]
        metadata = entry["metadata"]
        solution: list[list[int]] = metadata["solution"]
        board_size: int = len(solution[0])

        # 1. match answer without trailing whitespaces
        answer_stripped = "\n".join(l.rstrip() for l in answer.split("\n"))
        oracle_answer_stripped = "\n".join(l.rstrip() for l in oracle_answer.split("\n"))

        if answer_stripped == oracle_answer_stripped:
            reward = 1.0
        else:
            # 2. accept answers with correct numeric sequence (ignoring non-numeric characters)
            row = 0
            num_matching = 0
            for ln in answer.split("\n"):
                if row >= len(solution):
                    break
                numbers = [int(c) for c in ln if c in "123456789"]
                if len(numbers) != board_size:
                    continue  # ignore lines without numbers
                for a, b in zip(solution[row], numbers):
                    if a == b:
                        num_matching += 1
                row += 1

            reward = num_matching / (board_size * board_size)
            reward *= 0.9  # penalty for not using standard format

        if len(answer) > len(oracle_answer):
            reward *= len(oracle_answer) / len(answer)  # penalty for additional length
        return reward


register_dataset("mini_sudoku", MiniSudokuDataset, MiniSudokuConfig)
