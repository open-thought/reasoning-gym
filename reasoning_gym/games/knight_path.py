"""Knight's Shortest Path puzzle generator

Find the shortest sequence of knight moves between two squares on a chessboard,
navigating around optional blocked squares.
"""

import re
from collections import deque
from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from ..coaching import BaseCurriculum, RangeAttributeDefinition
from ..factory import ProceduralDataset, register_dataset

DATASET_NAME = "knight_path"

KNIGHT_MOVES = [
    (1, 2),
    (1, -2),
    (-1, 2),
    (-1, -2),
    (2, 1),
    (2, -1),
    (-2, 1),
    (-2, -1),
]

QUESTION_TEMPLATE = """You are given an {size}x{size} chessboard with a knight, a target square, and optional blocked squares.

The knight moves in an L-shape: 2 squares in one dimension and 1 square in the perpendicular dimension.
Knights can jump over other squares, but cannot land on blocked squares ('X').

Board:
{board}

- Knight start position (K): {start_square}
- Target square (#): {target_square}
- Blocked squares (X): {blocked_str}

Find the shortest sequence of moves for the knight to reach {target_square} from {start_square}.
Output the sequence of squares visited after {start_square}, separated by spaces (e.g. "c3 e4 g5").
If the target is unreachable, answer "infeasible".
"""


def _coord_to_square(r: int, c: int) -> str:
    """Convert (row, col) (0-indexed) to algebraic square notation, e.g. (3, 4) -> 'e4'"""
    file_char = chr(ord("a") + c)
    rank_str = str(r + 1)
    return f"{file_char}{rank_str}"


def _square_to_coord(sq: str) -> Optional[tuple[int, int]]:
    """Convert algebraic square notation like 'e4' to (row, col) (0-indexed)"""
    sq = sq.strip().lower()
    if len(sq) < 2 or not sq[0].isalpha() or not sq[1:].isdigit():
        return None
    c = ord(sq[0]) - ord("a")
    r = int(sq[1:]) - 1
    return (r, c)


@dataclass
class KnightPathConfig:
    """Configuration for Knight Path dataset generation"""

    board_size: int = 8
    min_obstacles: int = 0
    max_obstacles: int = 5
    allow_infeasible: bool = False
    size: int = 500
    seed: Optional[int] = None

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.size > 0, "size must be positive"
        assert 4 <= self.board_size <= 16, "board_size must be between 4 and 16"
        assert 0 <= self.min_obstacles, "min_obstacles must be non-negative"
        assert self.min_obstacles <= self.max_obstacles, "min_obstacles must be <= max_obstacles"
        max_possible_obstacles = (self.board_size * self.board_size) - 2
        assert self.max_obstacles <= max_possible_obstacles, "max_obstacles cannot exceed board capacity"


class KnightPathDataset(ProceduralDataset):
    """Generates Knight Shortest Path tasks with configurable obstacles and board size"""

    def __init__(self, config: KnightPathConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _render_board(
        self,
        board_size: int,
        start: tuple[int, int],
        target: tuple[int, int],
        blocked: set[tuple[int, int]],
    ) -> str:
        """Render board in standard chess perspective: rank board_size at top, rank 1 at bottom."""
        files_header = "  " + " ".join(chr(ord("a") + c) for c in range(board_size))
        lines = [files_header]

        for r in range(board_size - 1, -1, -1):
            rank_num = str(r + 1)
            row_cells = []
            for c in range(board_size):
                pos = (r, c)
                if pos == start:
                    row_cells.append("K")
                elif pos == target:
                    row_cells.append("#")
                elif pos in blocked:
                    row_cells.append("X")
                else:
                    row_cells.append(".")
            lines.append(f"{rank_num:>2} " + " ".join(row_cells) + f" {rank_num}")

        lines.append(files_header)
        return "\n".join(lines)

    def _bfs_shortest_path(
        self,
        board_size: int,
        start: tuple[int, int],
        target: tuple[int, int],
        blocked: set[tuple[int, int]],
    ) -> Optional[list[tuple[int, int]]]:
        """Find the shortest path from start to target avoiding blocked squares using BFS."""
        if start == target:
            return []

        queue = deque([(start[0], start[1], [])])
        visited = {start}

        while queue:
            r, c, path = queue.popleft()
            for dr, dc in KNIGHT_MOVES:
                nr, nc = r + dr, c + dc
                if 0 <= nr < board_size and 0 <= nc < board_size:
                    next_pos = (nr, nc)
                    if next_pos not in visited and next_pos not in blocked:
                        new_path = path + [next_pos]
                        if next_pos == target:
                            return new_path
                        visited.add(next_pos)
                        queue.append((nr, nc, new_path))
        return None

    def _generate_puzzle(self, rng: Random) -> dict[str, Any]:
        """Generate a single puzzle instance according to config."""
        board_size = self.config.board_size
        all_squares = [(r, c) for r in range(board_size) for c in range(board_size)]

        for _ in range(100):  # Retry attempts to satisfy allow_infeasible constraint
            rng.shuffle(all_squares)
            start = all_squares[0]
            target = all_squares[1]

            num_obs = rng.randint(self.config.min_obstacles, self.config.max_obstacles)
            blocked = set(all_squares[2 : 2 + num_obs])

            path = self._bfs_shortest_path(board_size, start, target, blocked)

            if path is not None or self.config.allow_infeasible:
                return {
                    "board_size": board_size,
                    "start": start,
                    "target": target,
                    "blocked": blocked,
                    "path": path,
                }

        # Fallback without obstacles if retry fails
        start, target = (0, 0), (board_size - 1, board_size - 1)
        blocked = set()
        path = self._bfs_shortest_path(board_size, start, target, blocked)
        return {
            "board_size": board_size,
            "start": start,
            "target": target,
            "blocked": blocked,
            "path": path,
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Generate a single Knight Path task"""
        rng = Random(self.seed + idx)
        puzzle = self._generate_puzzle(rng)

        board_size = puzzle["board_size"]
        start = puzzle["start"]
        target = puzzle["target"]
        blocked = puzzle["blocked"]
        path = puzzle["path"]

        start_sq = _coord_to_square(start[0], start[1])
        target_sq = _coord_to_square(target[0], target[1])
        blocked_squares = sorted([_coord_to_square(r, c) for r, c in blocked])
        blocked_str = ", ".join(blocked_squares) if blocked_squares else "None"

        board_str = self._render_board(board_size, start, target, blocked)

        if path is not None:
            answer_squares = [_coord_to_square(r, c) for r, c in path]
            answer_str = " ".join(answer_squares)
            shortest_distance = len(path)
        else:
            answer_str = "infeasible"
            shortest_distance = -1

        question = QUESTION_TEMPLATE.format(
            size=board_size,
            board=board_str,
            start_square=start_sq,
            target_square=target_sq,
            blocked_str=blocked_str,
        )

        return {
            "question": question,
            "answer": answer_str,
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "board_size": board_size,
                "start": start_sq,
                "target": target_sq,
                "blocked": blocked_squares,
                "shortest_distance": shortest_distance,
                "optimal_path": answer_str,
            },
        }

    def _parse_candidate_path(self, answer_str: str) -> list[str]:
        """Extract square tokens from candidate answer string."""
        # Clean delimiters: replace arrows, commas, brackets with spaces
        cleaned = re.sub(r"[->,\[\]]", " ", answer_str)
        tokens = [token.strip().lower() for token in cleaned.split() if token.strip()]
        return tokens

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """Verify the candidate knight path solution."""
        if not isinstance(answer, str) or len(answer.strip()) == 0:
            return 0.0

        oracle_answer = entry["answer"].strip()
        ans_clean = answer.strip().lower()

        # Handle infeasible case
        if oracle_answer == "infeasible":
            return 1.0 if "infeasible" in ans_clean else 0.0

        # Exact string match
        if ans_clean == oracle_answer.lower():
            return 1.0

        tokens = self._parse_candidate_path(ans_clean)
        if not tokens:
            return 0.0

        metadata = entry["metadata"]
        board_size = metadata["board_size"]
        start_sq = metadata["start"].lower()
        target_sq = metadata["target"].lower()
        blocked_set = set(sq.lower() for sq in metadata["blocked"])
        oracle_len = metadata["shortest_distance"]

        # If candidate starts with the start square, remove it
        if tokens[0] == start_sq:
            tokens = tokens[1:]

        if not tokens:
            return 0.0

        # Validate path
        current_coord = _square_to_coord(start_sq)
        if current_coord is None:
            return 0.0

        for sq_name in tokens:
            coord = _square_to_coord(sq_name)
            if coord is None:
                return 0.0
            r, c = coord
            if not (0 <= r < board_size and 0 <= c < board_size):
                return 0.0
            if sq_name in blocked_set:
                return 0.0
            # Must be a valid knight move from previous square
            dr = abs(r - current_coord[0])
            dc = abs(c - current_coord[1])
            if not ((dr == 1 and dc == 2) or (dr == 2 and dc == 1)):
                return 0.0
            current_coord = (r, c)

        # Must end at target
        if tokens[-1] != target_sq:
            return 0.0

        # If path is valid: full credit for optimal length, partial credit for suboptimal
        if len(tokens) == oracle_len:
            return 1.0
        elif len(tokens) > oracle_len:
            return 0.5

        return 0.0


class KnightPathCurriculum(BaseCurriculum):
    """Curriculum for progressive difficulty in Knight Shortest Path puzzles"""

    def __init__(self):
        super().__init__(KnightPathCurriculum.__name__, KnightPathConfig)
        self._define_attributes(
            RangeAttributeDefinition(
                name="max_obstacles",
                levels=[0, 2, 4, 6, 8],
                default_level=2,
                description="Maximum number of obstacle squares",
                lower_field_name="min_obstacles",
                upper_field_name="max_obstacles",
            ),
            RangeAttributeDefinition(
                name="board_size",
                levels=[6, 8, 10],
                default_level=1,
                description="Dimensions of the chessboard",
                lower_field_name="board_size",
                upper_field_name="board_size",
            ),
        )


register_dataset(DATASET_NAME, KnightPathDataset, KnightPathConfig, KnightPathCurriculum)
