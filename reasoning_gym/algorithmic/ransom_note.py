"""Check if you can construct a ransom note from letters in a magazine.

A popular Leetcode problem:
https://leetcode.com/problems/ransom-note/description/
"""

from collections import defaultdict
from dataclasses import dataclass
from random import Random
from typing import Optional

from ..coaching import AttributeType, BaseCurriculum, RangeAttributeDefinition
from ..factory import ProceduralDataset, register_dataset

QUESTION_TEMPLATE = """Given two strings representing a ransom note and a magazine, return True if you can construct the ransom note using the letters in the magazine, and False otherwise.

Each letter in the magazine string can only be used once in your ransom note.

Ransom note: {ransom_note}
Magazine: {magazine}
"""


@dataclass
class RansomNoteConfig:
    """Configuration for Ransom Note dataset generation"""

    min_note_length: int = 1  # Minimum length of the ransom note
    max_note_length: int = 10  # Maximum length of the ransom note
    min_magazine_length: int = 2  # Minimum length of the magazine
    max_magazine_length: int = 30  # Maximum length of the magazine
    p_solvable: float = 0.5  # Probability that the ransom note can be constructed

    size: int = 500  # Virtual dataset size
    seed: Optional[int] = None

    def validate(self):
        """Validate configuration parameters"""
        # assert 1 <= self.max_note_length <= MAX_NOTE_LENGTH, "max_note_length must be between 1 and MAX_NOTE_LENGTH"
        assert 1 <= self.min_note_length, "min_note_length must be at least 1"
        assert (
            self.min_note_length <= self.max_note_length
        ), "min_note_length must be less than or equal to max_note_length"
        assert 2 <= self.min_magazine_length, "min_magazine_length must be at least 2"
        assert (
            self.min_magazine_length <= self.max_magazine_length
        ), "min_magazine_length must be less than or equal to max_magazine_length"
        assert self.max_note_length < self.max_magazine_length, "max_note_length must be less than max_magazine_length"
        assert 0 <= self.p_solvable <= 1, "p_solvable must be between 0 and 1"


class RansomNoteDataset(ProceduralDataset):
    """Generates Ransom Note exercises with configurable difficulty"""

    def __init__(self, config: RansomNoteConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)
        self.letters = {chr(i) for i in range(ord("a"), ord("z") + 1)}

    def _get_inputs(self, rng: Random, note_length: int, magazine_length: int, solvable: bool) -> tuple[str, str]:
        """Generate random ransom note and magazine"""
        ransom_note = [rng.choice(list(self.letters)) for _ in range(note_length)]
        magazine = ransom_note.copy()
        if solvable:
            magazine.extend([rng.choice(list(self.letters)) for _ in range(magazine_length - note_length)])
        else:
            remove_letter = rng.choice(magazine)
            magazine.remove(remove_letter)
            magazine.extend(
                [rng.choice(list(self.letters - {remove_letter})) for _ in range(magazine_length - note_length + 1)]
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

        note_length = rng.randint(self.config.min_note_length, self.config.max_note_length)
        magazine_length = rng.randint(
            max(note_length, self.config.min_magazine_length), self.config.max_magazine_length
        )
        solvable = rng.random() < self.config.p_solvable
        ransom_note, magazine = self._get_inputs(rng, note_length, magazine_length, solvable)
        answer = self._can_construct(ransom_note, magazine)

        return {
            "question": QUESTION_TEMPLATE.format(ransom_note=ransom_note, magazine=magazine),
            "answer": str(answer),
            "metadata": {
                "ransom_note": ransom_note,
                "magazine": magazine,
                "solution": answer,
                "solvable": solvable,
                "difficulty": {
                    "note_length": note_length,
                    "magazine_length": magazine_length,
                },
            },
        }


class RansomNoteCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(RansomNoteCurriculum.__name__, RansomNoteConfig)

        # Define attributes
        self._define_attributes(
            RangeAttributeDefinition(
                name="note_length",
                levels=[10, 50, 100, 500],
                default_level=0,
                description="Length of the ransom note",
                attr_type=AttributeType.APPEND,
                min_value=1,
                lower_field_name="min_note_length",
                upper_field_name="max_note_length",
            ),
            RangeAttributeDefinition(
                name="magazine_length",
                levels=[50, 100, 500, 1000],
                default_level=0,
                description="Length of the magazine",
                attr_type=AttributeType.APPEND,
                min_value=2,
                lower_field_name="min_magazine_length",
                upper_field_name="max_magazine_length",
            ),
        )


register_dataset("ransom_note", RansomNoteDataset, RansomNoteConfig, RansomNoteCurriculum)
