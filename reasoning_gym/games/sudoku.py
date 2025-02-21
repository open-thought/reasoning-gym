"""Sudoku puzzle generator"""

import copy
from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from ..factory import ProceduralDataset, register_dataset


@dataclass
class SudokuConfig:
    """
    Configuration for sudoku puzzle generation
    Puzzle generation can be a bit slower for puzzles with a high (~60+) number of empty cells
    """

    min_empty: int = 30  # Minimum number of empty cells
    max_empty: int = 50  # Maximum number of empty cells
    seed: Optional[int] = None
    size: int = 500  # Virtual dataset size

    def validate(self):
        """Validate configuration parameters"""
        # 81 - 64 = 17, the minimum number of clues required for 9x9 Sudoku to have a unique solution
        assert 0 <= self.min_empty <= 64, "min_empty must be between 0 and 64"
        assert self.min_empty <= self.max_empty <= 64, "max_empty must be between min_empty and 64"


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

    def _is_valid(self, board: list[list[int]], row: int, col: int, num: int) -> bool:
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

    def _get_possible_values(self, board: list[list[int]], row: int, col: int) -> set[int]:
        """Get all possible values for a cell."""
        row_values = set(board[row])
        col_values = set(board[i][col] for i in range(9))

        # Get filled values in the current 3x3 subgrid
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        box_values = set()
        for i in range(box_row, box_row + 3):
            for j in range(box_col, box_col + 3):
                box_values.add(board[i][j])

        used_values = row_values | col_values | box_values
        return set(range(1, 10)) - used_values

    def _solve(self, board: list[list[int]]) -> bool:
        """Solve sudoku using backtracking"""
        empty = self._find_empty(board)
        if not empty:
            return True

        row, col = empty
        for num in self._get_possible_values(board, row, col):
            board[row][col] = num
            if self._solve(board):
                return True
            board[row][col] = 0
        return False

    def _find_empty(self, board: list[list[int]]) -> Optional[tuple[int, int]]:
        """Find an empty cell"""
        for i in range(9):
            for j in range(9):
                if board[i][j] == 0:
                    return (i, j)
        return None

    def _generate_solved_board(self, rng: Random) -> list[list[int]]:
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

    def _count_solutions(self, board: list[list[int]], limit: int = 2) -> int:
        """Count the number of solutions for a given board"""

        def _get_min_possibilities_cell(board: list[list[int]]) -> Optional[tuple[int, int, set[int]]]:
            """
            Get the cell with the lowest number of possibilities.
            Returns None if the board is already solved.
            """
            min_possibilities = 10
            min_cell = None
            min_values = None

            for i in range(9):
                for j in range(9):
                    if board[i][j] == 0:
                        possible = self._get_possible_values(board, i, j)
                        if len(possible) < min_possibilities:
                            min_possibilities = len(possible)
                            min_cell = (i, j)
                            min_values = possible
                            if min_possibilities == 1:
                                return (*min_cell, min_values)

            return (*min_cell, min_values) if min_cell else None

        def _count_solutions_helper(board: list[list[int]]) -> int:
            cell_info = _get_min_possibilities_cell(board)
            if not cell_info:
                return 1

            row, col, possible_values = cell_info
            count = 0
            for num in possible_values:
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
        cells = [(i, j) for i in range(9) for j in range(9)]
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

        question = (
            f"Solve this Sudoku puzzle:\n{puzzle_str}\n"
            "Respond with only your answer, formatted as the puzzle, a 9x9 grid with numbers separated by spaces, and rows separated by newlines."
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


register_dataset("sudoku", SudokuDataset, SudokuConfig)
