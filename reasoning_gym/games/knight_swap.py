import collections
import json
from dataclasses import dataclass
from random import Random
from typing import FrozenSet, Optional

from ..factory import ProceduralDataset, register_dataset

QUESTION_TEMPLATE = """Knight Swap Challenge:

```
{board}
```

Legend:
- 'w' = White Knight
- 'B' = Black Knight
- Empty squares are marked with '.'

Objective:
Swap the positions of all white knights with all black knights through valid moves.

Rules:
1. Knights move in L-shape (2 squares + 1 square perpendicular)
2. Knights can only move to empty squares
3. {start_turn} moves first, then players alternate
4. All knights must reach their target positions (white ↔ black)

Question:
Is it possible to swap all knights' positions? If yes, list the moves.

Answer Format:
- For impossible puzzles: "No"
- For possible puzzles: List moves as ["color,from,to", ...]
  Example: ["w,A1,B3"] means white knight moves A1→B3
"""


@dataclass
class KnightSwapConfig:
    """Configuration for Knight Swap puzzle generation.

    A Knight Swap puzzle involves moving white and black knights on a chess-like board
    where each move must be a valid knight's move. The goal is to swap the positions
    of white and black knights.
    """

    min_nodes: int = 6  # Minimum number of squares on the board
    max_nodes: int = 9  # Maximum number of squares on the board
    min_pieces: int = 2  # Minimum number of pieces per color
    max_pieces: int = 2  # Maximum number of pieces per color
    min_steps: int = 4  # Minimum solution length
    max_steps: int = 20  # Maximum solution length
    max_attempts: int = 100  # Maximum attempts for board generation and puzzle creation
    seed: Optional[int] = None
    size: int = 5  # Virtual dataset size
    impossible_ratio: float = 0.2  # Ratio of puzzles that should be impossible

    def validate(self):
        """Validate configuration parameters"""
        assert self.min_nodes >= 6, "min_nodes must be >= 6"
        assert self.max_nodes >= self.min_nodes, "max_nodes must be >= min_nodes"
        assert self.min_pieces >= 1, "min_pieces must be >= 1"
        assert self.max_pieces >= self.min_pieces, "max_pieces must be >= min_pieces"
        assert self.min_steps >= 1, "min_steps must be >= 1"
        assert self.max_steps >= self.min_steps, "max_steps must be >= min_steps"
        assert self.max_attempts >= 1, "max_attempts must be >= 1"
        assert 0 <= self.impossible_ratio <= 1, "impossible_ratio must be between 0 and 1"


class KnightSwapLogic:
    """Core game logic for Knight Swap puzzles."""

    @staticmethod
    def is_knight_move(a: str, b: str) -> bool:
        """Check if moving from square 'a' to square 'b' is a legal knight move."""
        a_col = ord(a[0].upper()) - ord("A") + 1
        a_row = int(a[1:])
        b_col = ord(b[0].upper()) - ord("A") + 1
        b_row = int(b[1:])
        return {abs(a_col - b_col), abs(a_row - b_row)} == {1, 2}

    @staticmethod
    def is_connected(graph: dict[str, list[str]]) -> bool:
        """Check if a graph is connected (all nodes reachable from any starting node)."""
        if not graph:
            return True
        start = next(iter(graph))
        visited = set()
        queue = collections.deque([start])
        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        queue.append(neighbor)
        return len(visited) == len(graph)

    @staticmethod
    def generate_board(num_nodes: int, rng: Random, max_attempts: int = 1000) -> dict[str, list[str]]:
        """Generate a random connected board where edges represent valid knight moves."""
        candidates = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3", "D1", "D2", "D3"]
        attempts = 0
        while True:
            attempts += 1
            nodes = rng.sample(candidates, num_nodes)
            graph = {node: [] for node in nodes}
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    if KnightSwapLogic.is_knight_move(nodes[i], nodes[j]):
                        graph[nodes[i]].append(nodes[j])
                        graph[nodes[j]].append(nodes[i])
            for node in graph:
                graph[node].sort()
            if KnightSwapLogic.is_connected(graph):
                return graph
            if attempts > max_attempts:
                raise Exception(f"Failed to generate connected board after {max_attempts} attempts")

    @staticmethod
    def solve_swap(
        board: dict[str, list[str]], pieces: dict[str, str], start_turn: str = "w"
    ) -> Optional[list[tuple[str, str, str]]]:
        """Find a sequence of moves to swap white and black pieces positions."""

        @dataclass(frozen=True)
        class GameState:
            white_set: FrozenSet[str]
            black_set: FrozenSet[str]
            turn: str

        initial_white = frozenset(pos for pos, piece in pieces.items() if piece == "w")
        initial_black = frozenset(pos for pos, piece in pieces.items() if piece == "B")
        initial_state = GameState(initial_white, initial_black, start_turn)

        queue = collections.deque([initial_state])
        visited = {initial_state}
        predecessors = {initial_state: (None, None)}

        while queue:
            state = queue.popleft()
            if state.white_set == initial_black and state.black_set == initial_white:
                moves = []
                cur_state = state
                while predecessors[cur_state][0] is not None:
                    prev_state, move = predecessors[cur_state]
                    moves.append(move)
                    cur_state = prev_state
                moves.reverse()
                return moves

            current_positions = state.white_set if state.turn == "w" else state.black_set
            for pos in current_positions:
                for neighbor in board[pos]:
                    if neighbor in state.white_set or neighbor in state.black_set:
                        continue
                    if state.turn == "w":
                        new_white = frozenset(p if p != pos else neighbor for p in state.white_set)
                        new_black = state.black_set
                    else:
                        new_black = frozenset(p if p != pos else neighbor for p in state.black_set)
                        new_white = state.white_set
                    next_turn = "B" if state.turn == "w" else "w"
                    new_state = GameState(new_white, new_black, next_turn)
                    if new_state not in visited:
                        visited.add(new_state)
                        predecessors[new_state] = (state, (state.turn, pos, neighbor))
                        queue.append(new_state)
        return None


class KnightSwapDataset(ProceduralDataset):
    """Generates Knight Swap puzzles with configurable parameters."""

    def __init__(self, config: KnightSwapConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)
        self.game_logic = KnightSwapLogic()

    def _format_board(self, board: dict[str, list[str]], pieces: dict[str, str]) -> str:
        """Format the board state as a string."""
        positions = list(board.keys())
        if not positions:
            return ""

        columns = sorted(set(pos[0] for pos in positions))
        rows = sorted(set(int(pos[1:]) for pos in positions), reverse=True)

        lines = []
        # Header
        lines.append("    " + "   ".join(columns))
        lines.append("   " + "----" * len(columns))

        # Board rows
        for row in rows:
            line = f"{row} |"
            for col in columns:
                pos = col + str(row)
                if pos in pieces:
                    piece = pieces[pos] if pieces[pos] is not None else "."
                    line += f" {piece} |"
                else:
                    line += "   |"
            lines.append(line)
            lines.append("   " + "----" * len(columns))

        return "\n".join(lines)

    def _format_moves(self, moves: list[tuple[str, str, str]]) -> str:
        """Format the solution moves as a string."""
        if not moves:
            return "No"
        return json.dumps([f"{color},{start},{end}" for color, start, end in moves])

    def __getitem__(self, idx: int) -> dict:
        """Generate a single Knight Swap puzzle."""
        rng = Random(self.seed + idx)

        # Keep trying with new boards until we succeed
        board_attempts = 0
        while board_attempts < self.config.max_attempts:
            try:
                # Generate a new board
                num_nodes = rng.randint(self.config.min_nodes, self.config.max_nodes)
                board = self.game_logic.generate_board(num_nodes, rng, max_attempts=self.config.max_attempts)
                positions = list(board.keys())

                # Decide if this should be an impossible puzzle
                make_impossible = rng.random() < self.config.impossible_ratio

                # Try different piece placements on this board
                for _ in range(50):  # Reduced attempts per board since we try multiple boards
                    # Use fixed number of pieces for more reliable generation
                    num_pieces = self.config.min_pieces
                    white_positions = rng.sample(positions, num_pieces)
                    remaining = [p for p in positions if p not in white_positions]
                    black_positions = rng.sample(remaining, num_pieces)

                    pieces = {pos: None for pos in positions}
                    for pos in white_positions:
                        pieces[pos] = "w"
                    for pos in black_positions:
                        pieces[pos] = "B"

                    # For impossible puzzles, try a simpler approach: just remove some key connections
                    board_copy = {k: list(v) for k, v in board.items()}  # Make a copy of the board
                    if make_impossible:
                        # Remove critical edges that would make the puzzle impossible
                        critical_edges = []
                        for w_pos in white_positions:
                            for b_pos in black_positions:
                                if b_pos in board_copy[w_pos]:
                                    critical_edges.append((w_pos, b_pos))

                        if critical_edges:  # Only proceed if we found critical edges
                            # Remove a random critical edge
                            w_pos, b_pos = rng.choice(critical_edges)
                            board_copy[w_pos].remove(b_pos)
                            board_copy[b_pos].remove(w_pos)

                    # Try both starting turns
                    for start_turn in ["w", "B"]:
                        solution = self.game_logic.solve_swap(board_copy, pieces, start_turn)

                        # Accept solutions with more flexible length requirements
                        if (make_impossible and solution is None) or (
                            not make_impossible
                            and solution is not None
                            and self.config.min_steps <= len(solution) <= self.config.max_steps
                        ):
                            board_str = self._format_board(board_copy, pieces)
                            solution_str = self._format_moves(solution) if solution else "No"

                            # Generate board states for solvable puzzles
                            board_states = []
                            if solution is not None:
                                current_pieces = dict(pieces)
                                board_states.append(dict(current_pieces))  # Initial state

                                for color, start, end in solution:
                                    current_pieces[end] = current_pieces[start]
                                    current_pieces[start] = None
                                    board_states.append(dict(current_pieces))

                            return {
                                "question": QUESTION_TEMPLATE.format(board=board_str, start_turn=start_turn),
                                "answer": solution_str,
                                "metadata": {
                                    "board": board_copy,
                                    "pieces": pieces,
                                    "start_turn": start_turn,
                                    "solution": solution,
                                    "is_possible": solution is not None,
                                    "num_steps": len(solution) if solution else 0,
                                    "board_states": board_states if solution is not None else None,
                                },
                            }

            except Exception:
                pass  # If board generation fails, we'll try again with a new board

            board_attempts += 1

        raise ValueError(f"Failed to generate valid puzzle after trying {self.config.max_attempts} different boards")

    def score_answer(self, answer: Optional[str], entry: dict) -> float:
        """Score the user's solution for the Knight Swap puzzle.

        The answer should be either:
        - "No" if the puzzle is impossible
        - A JSON list of moves in format ["color,start,end", ...] where color is 'w' or 'B'

        Returns:
        - 1.0 for correct answer (either "No" for impossible puzzles or valid solution of optimal length)
        - A proportional score for correct but longer solutions
        - 0.05 for valid moves that don't solve the puzzle
        - 0.01 for invalid format
        - 0.0 for None
        """
        if answer is None:
            return 0.0

        answer = answer.strip()
        if not answer:
            return 0.01

        # Handle impossible puzzles
        if not entry["metadata"]["is_possible"]:
            return 1.0 if answer.lower() == "no" else 0.01

        # Handle "No" answer for possible puzzles
        if answer.lower() == "no":
            return 0.01

        try:
            # Parse moves from JSON list
            move_list = json.loads(answer)
            if not isinstance(move_list, list):
                return 0.01

            # Parse moves
            moves = []
            for move_str in move_list:
                color, start, end = move_str.split(",")
                if color not in ("w", "B"):
                    return 0.01
                moves.append((color, start, end))

            # Validate and apply moves
            board = entry["metadata"]["board"]
            pieces = dict(entry["metadata"]["pieces"])
            current_turn = entry["metadata"]["start_turn"]

            # Track board states after each move
            board_states = []
            board_states.append(dict(pieces))  # Initial state

            for color, start, end in moves:
                if color != current_turn:
                    return 0.01
                if start not in pieces or pieces[start] != color:
                    return 0.01
                if end not in board[start]:
                    return 0.01
                if end in pieces and pieces[end] is not None:
                    return 0.01

                # Apply move
                pieces[end] = pieces[start]
                pieces[start] = None
                current_turn = "B" if current_turn == "w" else "w"

                # Store board state after this move
                board_states.append(dict(pieces))

            # Check if solved
            white_positions = {pos for pos, piece in pieces.items() if piece == "w"}
            black_positions = {pos for pos, piece in pieces.items() if piece == "B"}
            initial_white = {pos for pos, piece in entry["metadata"]["pieces"].items() if piece == "w"}
            initial_black = {pos for pos, piece in entry["metadata"]["pieces"].items() if piece == "B"}

            if white_positions == initial_black and black_positions == initial_white:
                optimal_moves = len(entry["metadata"]["solution"])
                # Add board states to metadata if solution is valid
                entry["metadata"]["board_states"] = board_states
                if len(moves) <= optimal_moves:
                    return 1.0
                else:
                    return optimal_moves / len(moves)
            return 0.05

        except Exception:
            return 0.01


register_dataset("knight_swap", KnightSwapDataset, KnightSwapConfig)
