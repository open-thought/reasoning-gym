
"""Codebreaker: a procedural Mastermind-style reasoning game."""

from dataclasses import dataclass
from itertools import product
from random import Random
from typing import Any, Optional

from ..dataset import ProceduralDataset
from ..factory import register_dataset

DATASET_NAME = "codebreaker"

DEFAULT_COLORS = ("RED", "BLUE", "GREEN", "YELLOW", "PURPLE", "ORANGE")


@dataclass
class CodebreakerConfig:
    """Configuration for Codebreaker puzzles."""

    code_length: int = 4
    colors: tuple[str, ...] = DEFAULT_COLORS
    allow_repeats: bool = True
    seed: Optional[int] = None
    size: int = 500

    def validate(self) -> None:
        """Validate the puzzle configuration."""
        if self.code_length < 2:
            raise ValueError("code_length must be at least 2")
        if len(self.colors) < 2:
            raise ValueError("At least two colors are required")
        if len(set(self.colors)) != len(self.colors):
            raise ValueError("Colors must be unique")
        if any(not color or not color.isalpha() for color in self.colors):
            raise ValueError("Colors must contain letters only")
        if not self.allow_repeats and self.code_length > len(self.colors):
            raise ValueError("Code length cannot exceed the number of colors")
        if self.size <= 0:
            raise ValueError("size must be positive")


def get_feedback(secret: tuple[str, ...], guess: tuple[str, ...]) -> tuple[int, int]:
    """Return (correct color and position, correct color wrong position)."""
    exact = sum(s == g for s, g in zip(secret, guess))

    secret_remaining = [
        color for s, color in zip(secret, secret) if False
    ]
    # Count unmatched colors without counting a position twice.
    unmatched_secret = [s for s, g in zip(secret, guess) if s != g]
    unmatched_guess = [g for s, g in zip(secret, guess) if s != g]

    from collections import Counter

    common = sum(
        (Counter(unmatched_secret) & Counter(unmatched_guess)).values()
    )
    return exact, common


def format_code(code: tuple[str, ...]) -> str:
    """Format a code as space-separated colors."""
    return " ".join(code)


class CodebreakerDataset(ProceduralDataset):
    """Generate Mastermind-style puzzles with a unique solution."""

    def __init__(
        self,
        config: CodebreakerConfig,
        seed: Optional[int] = None,
        size: Optional[int] = None,
    ):
        config.validate()
        super().__init__(
            config,
            seed=config.seed if seed is None else seed,
            size=config.size if size is None else size,
        )

    def _all_codes(self) -> list[tuple[str, ...]]:
        if self.config.allow_repeats:
            return list(product(self.config.colors, repeat=self.config.code_length))

        from itertools import permutations

        return list(permutations(self.config.colors, self.config.code_length))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0 or idx >= self.size:
            raise IndexError(idx)

        rng = Random(self.seed + idx)
        all_codes = self._all_codes()
        secret = rng.choice(all_codes)

        # Generate candidate guesses and retain clues that uniquely identify
        # the secret. The secret itself is never included as a guess.
        possible_guesses = [code for code in all_codes if code != secret]
        rng.shuffle(possible_guesses)

        clues: list[tuple[tuple[str, ...], int, int]] = []
        candidates = set(all_codes)

        for guess in possible_guesses:
            exact, misplaced = get_feedback(secret, guess)

            new_candidates = {
                candidate
                for candidate in candidates
                if get_feedback(candidate, guess) == (exact, misplaced)
            }

            if len(new_candidates) < len(candidates):
                clues.append((guess, exact, misplaced))
                candidates = new_candidates

            if len(candidates) == 1:
                break

        # Every code can be distinguished from all other codes by enough
        # guesses. This guard catches any unexpected generation problem.
        if candidates != {secret}:
            raise RuntimeError("Failed to generate a uniquely solvable puzzle")

        clue_lines = []
        for guess, exact, misplaced in clues:
            clue_lines.append(
                f"Guess: {format_code(guess)} | "
                f"Correct color and position: {exact} | "
                f"Correct color, wrong position: {misplaced}"
            )

        question = (
            f"Crack the secret code of {self.config.code_length} colors.\n"
            f"Available colors: {', '.join(self.config.colors)}.\n"
            f"Repeated colors are "
            f"{'allowed' if self.config.allow_repeats else 'not allowed'}.\n\n"
            "Feedback for each guess tells you how many colors are in the "
            "correct position and how many are present but in the wrong "
            "position. Each color occurrence can be counted only once.\n\n"
            "Clues:\n"
            + "\n".join(clue_lines)
            + "\n\nReturn only the secret code as space-separated color names."
        )

        return {
            "question": question,
            "answer": format_code(secret),
            "metadata": {
                "index": idx,
                "secret": secret,
                "clues": [
                    {
                        "guess": guess,
                        "exact": exact,
                        "misplaced": misplaced,
                    }
                    for guess, exact, misplaced in clues
                ],
                "colors": self.config.colors,
                "code_length": self.config.code_length,
                "allow_repeats": self.config.allow_repeats,
            },
        }

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """Score a response by validating the complete color sequence."""
        if not isinstance(answer, str):
            return 0.0

        normalized = " ".join(answer.strip().upper().split())
        expected = entry["answer"].upper()

        # Accept a response wrapped in a simple code fence or explanation
        # only when its final line contains the complete code.
        if normalized == expected:
            return 1.0

        for line in reversed(normalized.splitlines()):
            cleaned = line.strip(" `*\"'")
            if cleaned == expected:
                return 1.0

        return 0.0


register_dataset(DATASET_NAME, CodebreakerDataset, CodebreakerConfig)
