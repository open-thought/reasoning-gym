"""Go problem (tsumego) generator"""

import re
from dataclasses import dataclass
from random import Random
from typing import Any, Dict, List, Optional, Set, Tuple

from ..factory import ProceduralDataset, register_dataset

# Added constant to avoid repetition of adjacent directions
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


@dataclass
class TsumegoConfig:
    """Configuration for Tsumego problem generation"""

    min_board_size: int = 9
    max_board_size: int = 13
    max_stones: int = 15
    size: int = 100
    seed: Optional[int] = None

    def __post_init__(self):
        """Validate configuration parameters"""
        if self.min_board_size < 5:
            raise ValueError("min_board_size must be at least 5")
        if self.max_board_size > 19:
            raise ValueError("max_board_size must be at most 19")
        if self.min_board_size > self.max_board_size:
            raise ValueError("min_board_size must be less than or equal to max_board_size")
        if self.max_stones < 5:
            raise ValueError("max_stones must be at least 5")


class TsumegoDataset(ProceduralDataset):
    """Generates Tsumego problems with configurable parameters"""

    def __init__(self, config: TsumegoConfig):
        self._prompt_templates = [
            "Tsumego time. Black to play and capture some stones.\nFind the key move.",
            "I have a Go problem for you. Black moves next - can you capture some of the white stones?",
            "Here's a Go challenge. Playing as Black, how can you capture as many white stones as possible?",
        ]
        self._ko_point = None
        super().__init__(config=config, seed=config.seed, size=config.size)

    # New helper method for board copying
    def _copy_board(self, board: List[List[str]]) -> List[List[str]]:
        """Return a deep copy of the board."""
        return [row[:] for row in board]

    def _get_liberties(self, board: List[List[str]], row: int, col: int) -> Set[Tuple[int, int]]:
        """Get empty adjacent points (liberties) for a stone"""
        size = len(board)
        liberties = set()
        for dr, dc in DIRECTIONS:
            r, c = row + dr, col + dc
            if 0 <= r < size and 0 <= c < size and board[r][c] == ".":
                liberties.add((r, c))
        return liberties

    def _get_group(self, board: List[List[str]], row: int, col: int) -> Set[Tuple[int, int]]:
        """Get all stones in the same group (connected stones of same color)"""
        size = len(board)
        color = board[row][col]
        if color == ".":
            return set()

        group = {(row, col)}
        queue = [(row, col)]
        while queue:
            r, c = queue.pop(0)
            for dr, dc in DIRECTIONS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size and board[nr][nc] == color and (nr, nc) not in group:
                    group.add((nr, nc))
                    queue.append((nr, nc))
        return group

    def _count_liberties(self, board: List[List[str]], group: Set[Tuple[int, int]]) -> int:
        """Count total liberties for a group of stones"""
        liberties = set()
        for row, col in group:
            liberties.update(self._get_liberties(board, row, col))
        return len(liberties)

    def _would_capture(self, board: List[List[str]], row: int, col: int, color: str) -> bool:
        """Check if a move would capture any opponent stones"""
        size = len(board)
        opponent = "O" if color == "X" else "X"

        # Make a copy of the board and place the stone
        board_copy = self._copy_board(board)
        board_copy[row][col] = color

        checked = set()
        for dr, dc in DIRECTIONS:
            r, c = row + dr, col + dc
            if 0 <= r < size and 0 <= c < size and board_copy[r][c] == opponent and (r, c) not in checked:
                group = self._get_group(board_copy, r, c)
                checked.update(group)
                if self._count_liberties(board_copy, group) == 0:
                    return True
        return False

    def _is_valid_move(self, board: List[List[str]], row: int, col: int, color: str) -> bool:
        """Check if a move is legal (not suicide, unless it captures)"""
        size = len(board)
        if not (0 <= row < size and 0 <= col < size):
            return False
        if board[row][col] != ".":
            return False
        if (row, col) == self._ko_point:
            return False

        # If the move captures opponent stones, it's valid
        if self._would_capture(board, row, col, color):
            return True

        board_copy = self._copy_board(board)
        board_copy[row][col] = color
        group = self._get_group(board_copy, row, col)
        return self._count_liberties(board_copy, group) > 0

    def _make_move(self, board: List[List[str]], row: int, col: int, color: str) -> bool:
        """Make a move and update ko point. Returns True if move was valid."""
        if not self._is_valid_move(board, row, col, color):
            return False

        self._ko_point = None
        board[row][col] = color
        opponent = "O" if color == "X" else "X"
        captured_stones = []

        for dr, dc in DIRECTIONS:
            r, c = row + dr, col + dc
            if 0 <= r < len(board) and 0 <= c < len(board) and board[r][c] == opponent:
                group = self._get_group(board, r, c)
                if self._count_liberties(board, group) == 0:
                    captured_stones.extend(group)

        if len(captured_stones) == 1 and len(self._get_group(board, row, col)) == 1:
            self._ko_point = captured_stones[0]

        for r, c in captured_stones:
            board[r][c] = "."

        return True

    def _generate_capture_problem(self, size: int, rng: Random) -> Tuple[List[List[str]], Tuple[int, int]]:
        """Generate a capture problem"""
        board = [["." for _ in range(size)] for _ in range(size)]
        stones_placed = 0
        max_stones = self.config.max_stones - 4  # Reserve space for capture setup

        while stones_placed < max_stones:
            row = rng.randint(0, size - 1)
            col = rng.randint(0, size - 1)
            color = "X" if rng.random() < 0.5 else "O"
            if board[row][col] == "." and self._is_valid_move(board, row, col, color):
                self._make_move(board, row, col, color)
                stones_placed += 1

        tries = 0
        while tries < 50:
            row = rng.randint(1, size - 2)
            col = rng.randint(1, size - 2)
            capture_neighbors = [(0, 0)] + DIRECTIONS  # <-- incorporate (0,0) with the constant DIRECTIONS
            if board[row][col] == "." and all(board[row + dr][col + dc] == "." for dr, dc in capture_neighbors):
                board[row][col] = "O"
                board[row - 1][col] = "O"
                board[row + 1][col] = "O"
                board[row][col - 1] = "O"
                if self._is_valid_move(board, row, col + 1, "X"):
                    return board, (row, col + 1)
            tries += 1
        raise RuntimeError("Failed to generate a capture problem")

    def _board_to_string(self, board: List[List[str]]) -> str:
        """Convert board to string representation"""
        size = len(board)
        # Column labels
        cols = "   " + " ".join(chr(ord("A") + i) for i in range(size)) + "\n"
        # Board with row numbers
        rows = [f"{size-i:2d} {' '.join(row)}" for i, row in enumerate(board)]
        return cols + "\n".join(rows)

    def __getitem__(self, idx: int) -> dict:
        """Generate a single Tsumego problem

        Returns:
            dict with:
            - "question": Problem description and board state
            - "answer": Solution move(s)
            - "metadata": Problem details and configuration
        """
        rng = Random(self.seed + idx if self.seed is not None else None)
        size = rng.randint(self.config.min_board_size, self.config.max_board_size)

        board, solution = self._generate_capture_problem(size, rng)
        board_str = self._board_to_string(board)
        solution_str = f"{chr(ord('A')+solution[1])}{size-solution[0]}"

        return {
            "question": (
                rng.choice(self._prompt_templates) + "\n\n" + board_str + "\n\n"
                "X - Black\n"
                "O - White\n\n"
                "Specify your move in coordinates (e.g. 'C4' for column C, row 4)"
            ),
            "answer": solution_str,
            "metadata": {
                "difficulty": {"board_size": size},
                "board": board,
                "solution": solution,
            },
        }

    def score_answer(self, answer: Optional[str], entry: Dict[str, Any]) -> float:
        """Score the answer against the solution"""
        if answer is None:
            return 0.0
        answer = answer.strip()
        if not answer:
            return 0.01
        metadata = entry["metadata"]
        board_size = len(metadata["board"])
        expected_row, expected_col = metadata["solution"]  # get solution from (row, col) tuple

        try:
            # Assume letter-number format, e.g. "C4"
            m = re.match(r"^([A-Za-z])(\d+)$", answer)
            if not m:
                return 0.01
            col_letter, row_str = m.group(1), m.group(2)
            row = board_size - int(row_str)
            col = ord(col_letter.upper()) - ord("A")
            if (row, col) == (expected_row, expected_col):
                return 1.0

            if 0 <= row < board_size and 0 <= col < board_size:
                return 0.05
        except Exception:
            return 0.01
        return 0.01


# Register the dataset
register_dataset("tsumego", TsumegoDataset, TsumegoConfig)
