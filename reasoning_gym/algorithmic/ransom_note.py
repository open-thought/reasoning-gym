"""Check if you can construct a ransom note from letters in a magazine.

A popular Leetcode problem:
https://leetcode.com/problems/ransom-note/description/
"""

from collections import defaultdict
from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from ..factory import ProceduralDataset, register_dataset

MAX_NOTE_LENGTH = 100_000
MAX_MAGAZINE_LENGTH = 100_001

QUESTION_TEMPLATE = """Given two strings representing a ransom note and a magazine, return True if you can construct the ransom note using the letters in the magazine, and False otherwise.

Each letter in the magazine string can only be used once in your ransom note.

Ransom note: {ransom_note}
Magazine: {magazine}
"""


@dataclass
class RansomNoteConfig:
    """Configuration for Ransom Note dataset generation"""

    max_note_length: int = 10  # Maximum length of the ransom note
    max_magazine_length: int = 30  # Maximum length of the magazine
    p_solvable: float = 0.5  # Probability that the ransom note can be constructed

    size: int = 500  # Virtual dataset size
    seed: Optional[int] = None

    def validate(self):
        """Validate configuration parameters"""
        assert 1 <= self.max_note_length <= MAX_NOTE_LENGTH, "max_note_length must be between 1 and MAX_NOTE_LENGTH"
        assert (
            2 <= self.max_magazine_length <= MAX_MAGAZINE_LENGTH
        ), "max_magazine_length must be between 2 and MAX_MAGAZINE_LENGTH"
        assert self.max_note_length < self.max_magazine_length, "max_note_length must be less than max_magazine_length"
        assert 0 <= self.p_solvable <= 1, "p_solvable must be between 0 and 1"


class RansomNoteDataset(ProceduralDataset):
    """Generates Ransom Note exercises with configurable difficulty"""

    def __init__(self, config: RansomNoteConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)
        self.letters = {chr(i) for i in range(ord("a"), ord("z") + 1)}

    def _get_inputs(self, rng: Random, solvable: bool) -> tuple[str, str]:
        """Generate random ransom note and magazine"""
        ransom_note_len = rng.randint(1, self.config.max_note_length)
        ransom_note = [rng.choice(list(self.letters)) for _ in range(ransom_note_len)]

        magazine_len = rng.randint(ransom_note_len, self.config.max_magazine_length)
        magazine = ransom_note.copy()
        if solvable:
            magazine.extend([rng.choice(list(self.letters)) for _ in range(magazine_len - ransom_note_len)])
        else:
            remove_letter = rng.choice(magazine)
            magazine.remove(remove_letter)
            magazine.extend(
                [rng.choice(list(self.letters - {remove_letter})) for _ in range(magazine_len - ransom_note_len + 1)]
            )

        rng.shuffle(ransom_note)
        rng.shuffle(magazine)
        return "".join(ransom_note), "".join(magazine)

    def _can_construct(self, ransom_note: str, magazine: str) -> bool:
        """Check if ransom note can be constructed from magazine"""
        count = defaultdict(int)
        for c in magazine:
            count[c] += 1
        for c in ransom_note:
            if count[c] <= 0:
                return False
            count[c] -= 1
        return True

    def __getitem__(self, idx: int) -> dict:
        """Generate a single Group Anagrams question"""
        rng = Random(self.seed + idx)
        solvable = rng.random() < self.config.p_solvable
        ransom_note, magazine = self._get_inputs(rng, solvable)
        answer = self._can_construct(ransom_note, magazine)

        return {
            "question": QUESTION_TEMPLATE.format(ransom_note=ransom_note, magazine=magazine),
            "answer": str(answer),
            "metadata": {"ransom_note": ransom_note, "magazine": magazine, "solution": answer, "solvable": solvable},
        }

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """Determine if the solution provided solves this task.

        The function awards 1.0 for a correct answer.

        Args:
            answer (Optional[str]): The user's answer.
            entry (dict[str, Any]): The original dataset entry containing the correct answer.

        Returns:
            float: The computed score between 0.0 and 1.0.
        """

        if answer == None:
            return 0.0

        s_answer = answer.strip()
        if not s_answer == str(entry["answer"]):
            return 0.01
        else:
            return 1.0


register_dataset("ransom_note", RansomNoteDataset, RansomNoteConfig)
