"""Go problem (tsumego) generator"""

"""
This module generates one-move Tsumego puzzles, which are Go problems focused on tactical capture scenarios.

The puzzles generated here have the following characteristics:
- They are created on a board of configurable size (with a minimum and maximum board size).
- A number of stones are randomly placed on the board, subject to a maximum stone limit.
- A specific capture problem is then constructed by arranging white stones in a plus-shaped formation.
- Extra liberties surrounding this white group are filled with black stones, except for one key liberty.
  This forces a situation where a single move by Black (at the remaining liberty) results in a capture.
- Puzzle generation is deterministic given a seed, which ensures reproducibility.

These puzzles are intended to provide focused practice on reading and executing capturing moves in Go.

TODO: Generate multi-step Tsumego problems.
"""

import re
from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from ..factory import ProceduralDataset, register_dataset

# Added constant to avoid repetition of adjacent directions
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


@dataclass
class TsumegoConfig:
    """Configuration for Tsumego problem generation"""

    min_board_size: int = 9
    max_board_size: int = 13
    max_stones: int = 15
    size: int = 500
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
    def _copy_board(self, board: list[list[str]]) -> list[list[str]]:
        """Return a deep copy of the board."""
        return [row[:] for row in board]

    def _get_liberties(self, board: list[list[str]], row: int, col: int) -> set[tuple[int, int]]:
        """Get empty adjacent points (liberties) for a stone"""
        size = len(board)
        liberties = set()
        for dr, dc in DIRECTIONS:
            r, c = row + dr, col + dc
            if 0 <= r < size and 0 <= c < size and board[r][c] == ".":
                liberties.add((r, c))
        return liberties

    def _get_group(self, board: list[list[str]], row: int, col: int) -> set[tuple[int, int]]:
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

    def _count_liberties(self, board: list[list[str]], group: set[tuple[int, int]]) -> int:
        """Count total liberties for a group of stones"""
        liberties = set()
        for row, col in group:
            liberties.update(self._get_liberties(board, row, col))
        return len(liberties)

    def _would_capture(self, board: list[list[str]], row: int, col: int, color: str) -> bool:
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

    def _is_valid_move(self, board: list[list[str]], row: int, col: int, color: str) -> bool:
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

    def _make_move(self, board: list[list[str]], row: int, col: int, color: str) -> bool:
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

    def _generate_capture_problem(self, size: int, rng: Random) -> tuple[list[list[str]], tuple[int, int]]:
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
        formation_options = {
            "plus": {
                "white_offsets": [(0, 0), (-1, 0), (1, 0), (0, -1)],
                "forced_move_offset": (0, 1),
                "neighbor_offsets": [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)],
            },
            "L": {
                "white_offsets": [(0, 0), (0, 1), (1, 0)],
                "forced_move_offset": (1, 1),
                "neighbor_offsets": [(0, 0), (0, 1), (1, 0), (1, 1)],
            },
            "T": {
                "white_offsets": [(0, -1), (0, 0), (0, 1), (1, 0)],
                "forced_move_offset": (-1, 0),
                "neighbor_offsets": [(0, -1), (0, 0), (0, 1), (1, 0), (-1, 0)],
            },
        }

        while tries < 50:
            row = rng.randint(1, size - 2)
            col = rng.randint(1, size - 2)
            formation_type = rng.choice(list(formation_options.keys()))
            formation = formation_options[formation_type]
            if all(board[row + dr][col + dc] == "." for dr, dc in formation["neighbor_offsets"]):
                # Place white stones according to chosen formation
                for dr, dc in formation["white_offsets"]:
                    board[row + dr][col + dc] = "O"
                forced_move = (row + formation["forced_move_offset"][0], col + formation["forced_move_offset"][1])
                white_group = {(row + dr, col + dc) for dr, dc in formation["white_offsets"]}
                extra_liberties = set()
                for r, c in white_group:
                    extra_liberties |= self._get_liberties(board, r, c)
                extra_liberties.discard(forced_move)
                for r, c in extra_liberties:
                    board[r][c] = "X"

                # Add decoy stone to enhance puzzle difficulty
                current_stone_count = sum(cell in "XO" for row in board for cell in row)
                if current_stone_count < self.config.max_stones + 7:
                    center = (row, col)  # using the base white stone as center
                    decoy_candidates = []
                    for i in range(center[0] - 2, center[0] + 3):
                        for j in range(center[1] - 2, center[1] + 3):
                            if abs(i - center[0]) + abs(j - center[1]) == 2:
                                if 0 <= i < size and 0 <= j < size and board[i][j] == "." and (i, j) != forced_move:
                                    decoy_candidates.append((i, j))
                    if decoy_candidates:
                        decoy_pos = rng.choice(decoy_candidates)
                        decoy_color = "X" if rng.random() < 0.5 else "O"
                        board[decoy_pos[0]][decoy_pos[1]] = decoy_color

                if self._is_valid_move(board, forced_move[0], forced_move[1], "X"):
                    return board, forced_move
            tries += 1
        raise RuntimeError("Failed to generate a capture problem")

    def _board_to_string(self, board: list[list[str]]) -> str:
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
        solution_str = f"{chr(ord('A')+solution[1])}{size - solution[0]}"
        self._ko_point = None

        return {
            "question": (
                rng.choice(self._prompt_templates) + "\n\n" + board_str + "\n\n"
                "X - Black\n"
                "O - White\n\n"
                "Specify your move in coordinates (e.g. 'C4' for column C, row 4)"
            ),
            "answer": solution_str,
            "metadata": {"difficulty": {"board_size": size}, "board": board, "solution": solution_str},
        }

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
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
