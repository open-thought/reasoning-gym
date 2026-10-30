"""Pyraminx reasoning environment for procedural dataset generation and RLVR.

A Pyraminx is a regular tetrahedron twisty puzzle invented by Uwe Meffert.
It consists of 4 faces, 4 vertex tips, 4 vertex centers (axial pieces), and 6 edges,
comprising a total of 36 triangular stickers (9 per face).
"""

import re
from collections import deque
from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from ..coaching import BaseCurriculum, RangeAttributeDefinition, ScalarAttributeDefinition
from ..factory import ProceduralDataset, register_dataset

DATASET_NAME = "pyraminx"

# Face identifiers and default face colors
# 0: Front (G = Green), 1: Left (R = Red), 2: Right (B = Blue), 3: Down (Y = Yellow)
FACE_NAMES = ["F", "L", "R", "D"]
DEFAULT_COLORS = ["G", "R", "B", "Y"]

# Helper to index sticker positions: face in 0..3, pos in 0..8
def _idx(face: int, pos: int) -> int:
    return face * 9 + pos


# Clockwise 3-cycles (a, b, c) where sticker at a -> b -> c -> a.
# In state tuple: new_state[b] = old_state[a], new_state[c] = old_state[b], new_state[a] = old_state[c].
MOVE_CYCLES = {
    # Tip moves (lowercase): rotate only the small 1-piece tip at each vertex
    "u": [(_idx(0, 0), _idx(1, 0), _idx(2, 0))],
    "l": [(_idx(0, 4), _idx(3, 4), _idx(1, 8))],
    "r": [(_idx(0, 8), _idx(2, 4), _idx(3, 8))],
    "b": [(_idx(1, 4), _idx(3, 0), _idx(2, 8))],
    # Vertex layer moves (uppercase): rotate vertex tip, center, and 3 adjacent edges
    "U": [
        (_idx(0, 0), _idx(1, 0), _idx(2, 0)),  # tip U
        (_idx(0, 2), _idx(1, 2), _idx(2, 2)),  # center U
        (_idx(0, 1), _idx(1, 1), _idx(2, 1)),  # edge sticker 1
        (_idx(1, 3), _idx(2, 3), _idx(0, 3)),  # edge sticker 2
    ],
    "L": [
        (_idx(0, 4), _idx(3, 4), _idx(1, 8)),  # tip L
        (_idx(0, 5), _idx(3, 5), _idx(1, 7)),  # center L
        (_idx(0, 1), _idx(3, 6), _idx(1, 6)),  # edge sticker 1
        (_idx(1, 3), _idx(0, 6), _idx(3, 1)),  # edge sticker 2
    ],
    "R": [
        (_idx(0, 8), _idx(2, 4), _idx(3, 8)),  # tip R
        (_idx(0, 7), _idx(2, 5), _idx(3, 7)),  # center R
        (_idx(0, 3), _idx(2, 6), _idx(3, 6)),  # edge sticker 1
        (_idx(2, 1), _idx(3, 3), _idx(0, 6)),  # edge sticker 2
    ],
    "B": [
        (_idx(1, 4), _idx(3, 0), _idx(2, 8)),  # tip B
        (_idx(1, 5), _idx(3, 2), _idx(2, 7)),  # center B
        (_idx(1, 1), _idx(3, 1), _idx(2, 6)),  # edge sticker 1
        (_idx(2, 3), _idx(1, 6), _idx(3, 3)),  # edge sticker 2
    ],
}

ALL_LAYER_MOVES = ["U", "U'", "L", "L'", "R", "R'", "B", "B'"]
ALL_TIP_MOVES = ["u", "u'", "l", "l'", "r", "r'", "b", "b'"]


class PyraminxState:
    """Represents the internal state of a Pyraminx puzzle (36 stickers)."""

    def __init__(self, stickers: Optional[tuple[str, ...]] = None):
        if stickers is None:
            # Solved state: 9 stickers per face with face color
            self.stickers = tuple(DEFAULT_COLORS[f] for f in range(4) for _ in range(9))
        else:
            self.stickers = tuple(stickers)

    @property
    def solved_state(self) -> tuple[str, ...]:
        return tuple(DEFAULT_COLORS[f] for f in range(4) for _ in range(9))

    def is_solved(self) -> bool:
        # Check if each face has all 9 stickers of the same color
        for f in range(4):
            face_stickers = self.stickers[f * 9 : (f + 1) * 9]
            target_color = face_stickers[2]  # center sticker of this face
            if any(s != target_color for s in face_stickers):
                return False
        return True

    def apply_move(self, move: str) -> "PyraminxState":
        """Apply a single move (e.g. U, U', u, u') and return a new state."""
        base = move[0]
        is_prime = move.endswith("'")
        if base not in MOVE_CYCLES:
            raise ValueError(f"Unknown move: {move}")

        cycles = MOVE_CYCLES[base]
        new_stickers = list(self.stickers)
        for a, b, c in cycles:
            if not is_prime:
                # Clockwise: a -> b -> c -> a
                new_stickers[b] = self.stickers[a]
                new_stickers[c] = self.stickers[b]
                new_stickers[a] = self.stickers[c]
            else:
                # Counter-clockwise (prime): a -> c -> b -> a
                new_stickers[c] = self.stickers[a]
                new_stickers[b] = self.stickers[c]
                new_stickers[a] = self.stickers[b]

        return PyraminxState(tuple(new_stickers))

    def apply_moves(self, move_sequence: list[str]) -> "PyraminxState":
        current = self
        for m in move_sequence:
            current = current.apply_move(m)
        return current

    def partial_score(self) -> float:
        """Fraction of stickers matching the center color of their face."""
        correct = 0
        for f in range(4):
            face_stickers = self.stickers[f * 9 : (f + 1) * 9]
            target_color = face_stickers[2]  # face center
            correct += sum(1 for s in face_stickers if s == target_color)
        return correct / 36.0

    def render_ascii(self) -> str:
        """Render the 4 triangular faces into clear ASCII diagrams."""
        face_labels = [
            ("F (Front)", 0),
            ("L (Left)", 1),
            ("R (Right)", 2),
            ("D (Down/Bottom)", 3),
        ]
        sections = []
        for name, f in face_labels:
            st = self.stickers[f * 9 : (f + 1) * 9]
            # Row 0: 1 triangle [0]
            # Row 1: 3 triangles [1] [2] [3]
            # Row 2: 5 triangles [4] [5] [6] [7] [8]
            art = (
                f"Face {name}:\n"
                f"        [{st[0]}]\n"
                f"    [{st[1]}] [{st[2]}] [{st[3]}]\n"
                f"[{st[4]}] [{st[5]}] [{st[6]}] [{st[7]}] [{st[8]}]"
            )
            sections.append(art)
        return "\n\n".join(sections)


def solve_pyraminx(start_state: PyraminxState, max_bfs_depth: int = 7) -> list[str]:
    """Find a verified solution sequence for a Pyraminx state.

    First solves the 4 tips in O(1), then uses BFS to find the optimal layer solution.
    """
    if start_state.is_solved():
        return []

    solution_moves: list[str] = []
    current = start_state

    # 1. Solve the 4 tips independently (O(1))
    tip_checks = [
        ("u", _idx(0, 0), _idx(0, 2)),  # Tip U: matches face F center
        ("l", _idx(0, 4), _idx(0, 5)),  # Tip L: matches face F bottom-left center
        ("r", _idx(0, 8), _idx(0, 7)),  # Tip R: matches face F bottom-right center
        ("b", _idx(1, 4), _idx(1, 5)),  # Tip B: matches face L bottom-left center
    ]

    for tip_base, tip_idx, center_idx in tip_checks:
        for m in ["", tip_base, f"{tip_base}'"]:
            test_st = current if not m else current.apply_move(m)
            if test_st.stickers[tip_idx] == test_st.stickers[center_idx]:
                if m:
                    solution_moves.append(m)
                    current = test_st
                break

    if current.is_solved():
        return solution_moves

    # 2. BFS for the main body
    queue = deque([(current, [])])
    visited = {current.stickers}

    while queue:
        st, path = queue.popleft()
        if len(path) >= max_bfs_depth:
            break
        last_base = path[-1][0] if path else ""
        for m in ALL_LAYER_MOVES:
            if m[0] == last_base:
                continue
            nxt = st.apply_move(m)
            if nxt.is_solved():
                return solution_moves + path + [m]
            if nxt.stickers not in visited:
                visited.add(nxt.stickers)
                queue.append((nxt, path + [m]))

    return solution_moves


@dataclass
class PyraminxConfig:
    """Configuration for Pyraminx task generation."""

    min_scramble_steps: int = 2
    max_scramble_steps: int = 6
    include_tips: bool = True
    seed: Optional[int] = None
    size: int = 500

    def validate(self) -> None:
        """Validate configuration parameters."""
        assert self.min_scramble_steps > 0, "min_scramble_steps must be > 0"
        assert self.max_scramble_steps >= self.min_scramble_steps, "max_scramble_steps must be >= min_scramble_steps"
        assert self.size > 0, "size must be > 0"


class PyraminxDataset(ProceduralDataset):
    """Generates Pyraminx reasoning and solving tasks."""

    def __init__(self, config: PyraminxConfig):
        self._prompt_templates = [
            "You are given a scrambled Pyraminx (a 4-sided tetrahedral twisty puzzle). It has four triangular faces: Front (F), Left (L), Right (R), and Down (D), each consisting of 9 triangular stickers with colors Green (G), Red (R), Blue (B), and Yellow (Y).\n\nThe current puzzle state is:\n\n{puzzle_render}\n\nPlease provide a sequence of moves to solve this Pyraminx. Use standard WCA Pyraminx notation with space-separated moves (e.g. 'U R\\' L U\\''). Upper-case letters (U, L, R, B) turn vertex layers, and lower-case letters (u, l, r, b) turn vertex tips. Counter-clockwise turns are indicated with an apostrophe (').",
            "You see a Pyraminx puzzle arranged as follows:\n\n{puzzle_render}\n\nProvide the move sequence to solve the puzzle in standard Pyraminx notation. Do not combine steps, and separate all moves with spaces (for example: 'R U R\\' U\\'').",
        ]
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _generate_scramble(self, rng: Random, num_steps: int) -> list[str]:
        moves = []
        last_base = ""
        # Main layer moves
        for _ in range(num_steps):
            choices = [m for m in ALL_LAYER_MOVES if m[0] != last_base]
            m = rng.choice(choices)
            moves.append(m)
            last_base = m[0]

        # Optional tip scramble
        if self.config.include_tips:
            for tip in ["u", "l", "r", "b"]:
                if rng.random() < 0.6:
                    tip_turn = rng.choice([tip, f"{tip}'"])
                    moves.append(tip_turn)

        return moves

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Generate a single Pyraminx task."""
        rng = Random(self.seed + idx)

        num_steps = rng.randint(self.config.min_scramble_steps, self.config.max_scramble_steps)
        scramble_moves = self._generate_scramble(rng, num_steps)

        state = PyraminxState()
        state = state.apply_moves(scramble_moves)

        # Solve to get an example correct answer
        solution_moves = solve_pyraminx(state, max_bfs_depth=max(num_steps + 1, 7))
        if not solution_moves:
            # Fallback: inverse of scramble
            inverse_moves = []
            for m in reversed(scramble_moves):
                inv = m[0] if m.endswith("'") else f"{m}'"
                inverse_moves.append(inv)
            solution_moves = inverse_moves

        puzzle_render = state.render_ascii()
        prompt = rng.choice(self._prompt_templates).format(puzzle_render=puzzle_render)

        return {
            "question": prompt,
            "answer": " ".join(solution_moves),
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "scramble_steps": num_steps,
                "scramble_moves": " ".join(scramble_moves),
                "example_correct_answer": " ".join(solution_moves),
                "difficulty": {
                    "scramble_steps": (self.config.min_scramble_steps, self.config.max_scramble_steps),
                    "include_tips": self.config.include_tips,
                },
            },
        }

    def expand_moves(self, move_str: str) -> list[str]:
        """Normalize and parse move string into valid individual moves."""
        # Normalize quotes and delimiters
        cleaned = move_str.replace("’", "'").replace("`", "'").replace(",", " ")
        tokens = cleaned.strip().split()
        expanded = []
        valid_bases = {"U", "L", "R", "B", "u", "l", "r", "b"}

        for token in tokens:
            match = re.fullmatch(r"^([ULRBulrb])(['`]?)([23]?)$", token)
            if not match:
                continue
            base, prime, count = match.groups()
            rep = int(count) if count else 1
            if rep == 2:
                # In Pyraminx (order 3): M2 == M'
                expanded.append(f"{base}'" if not prime else base)
            elif rep == 3:
                # M3 == Identity
                continue
            else:
                expanded.append(f"{base}'" if prime else base)

        return expanded

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """Score the candidate answer by applying moves and verifying puzzle state."""
        if answer is None or len(answer.strip()) == 0:
            return 0.0

        try:
            # Reconstruct initial scrambled state
            eval_state = PyraminxState()
            scramble_moves = entry["metadata"]["scramble_moves"].split()
            eval_state = eval_state.apply_moves(scramble_moves)

            # Apply candidate moves
            candidate_moves = self.expand_moves(answer)
            if not candidate_moves:
                return 0.01

            eval_state = eval_state.apply_moves(candidate_moves)

            if eval_state.is_solved():
                return 1.0

            # Partial credit based on correctly solved stickers
            partial = eval_state.partial_score()
            return max(0.05, partial)

        except Exception:
            return 0.01


class PyraminxCurriculum(BaseCurriculum):
    """Curriculum for Pyraminx task difficulty progression."""

    def __init__(self):
        super().__init__(PyraminxCurriculum.__name__, PyraminxConfig)
        self._define_attributes(
            RangeAttributeDefinition(
                name="scramble_steps",
                levels=[2, 4, 6, 8, 12],
                description="Number of random moves to scramble the Pyraminx",
                lower_field_name="min_scramble_steps",
                upper_field_name="max_scramble_steps",
                ensure_interval=True,
            ),
            ScalarAttributeDefinition(
                name="include_tips",
                field_name="include_tips",
                levels=[False, True],
                description="Whether to scramble and require solving vertex tips",
            ),
        )


# Register dataset with Reasoning Gym
register_dataset(DATASET_NAME, PyraminxDataset, PyraminxConfig, PyraminxCurriculum)
