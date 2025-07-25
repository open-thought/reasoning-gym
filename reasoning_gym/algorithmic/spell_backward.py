"""Spell backward task generator"""

import re
from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from ..coaching import BaseCurriculum, RangeAttributeDefinition
from ..data import read_data_file
from ..factory import ProceduralDataset, register_dataset

DATASET_NAME = "spell_backward"


@dataclass
class SpellBackwardConfig:
    """Configuration for spelling words backward task generation"""

    min_word_len: int = 3  # Minimum word length
    max_word_len: int = 10  # Maximum word length
    seed: Optional[int] = None
    data_file: str = "words3to10.txt"
    size: int = 500  # Virtual dataset size

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.min_word_len > 0, "min_word_len must be positive"
        assert self.max_word_len >= self.min_word_len, "max_word_len must be >= min_word_len"


class SpellBackwardDataset(ProceduralDataset):
    """Generates tasks to spell words backward"""

    def __init__(self, config: SpellBackwardConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

        # Load and preprocess text
        text = read_data_file(self.config.data_file)
        self.words = [
            word.strip()
            for word in text.splitlines()
            if word.strip().isalnum() and config.min_word_len <= len(word.strip()) <= config.max_word_len
        ]

    def __getitem__(self, idx: int) -> dict:
        """Generate a single spell backward task"""
        rng = Random(self.seed + idx)

        # Select random word
        word = rng.choice(self.words)
        answer = word[::-1]

        return {
            "question": f"Spell this word backward (example: sun -> nus): {word}",
            "answer": answer,
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "word": word,
                "word_len": len(word),
                "difficulty": {
                    "word_len": (self.config.min_word_len, self.config.max_word_len),
                },
            },
        }

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        reward = 0.0
        expected_answer = entry["answer"]
        if isinstance(answer, str):
            try:
                expected_answer = expected_answer.lower()
                answer = answer.lower()
                if expected_answer == answer:
                    return 1.0
                else:
                    answer_len = len(expected_answer)
                    for i in range(len(expected_answer)):
                        if i < len(expected_answer) and i < len(answer):
                            if expected_answer[i] == answer[i]:
                                reward += 1 / answer_len
                            else:
                                continue
                        else:
                            break
                    if reward == 1.0:
                        reward -= 0.2
            except:
                reward = 0.0
        return reward


class SpellBackwardCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(SpellBackwardCurriculum.__name__, SpellBackwardConfig)

        # Define attributes
        self._define_attributes(
            RangeAttributeDefinition(
                name="word_len",
                levels=list(range(3, 13, 2)),
                description="Word length",
                lower_field_name="min_word_len",
                upper_field_name="max_word_len",
                ensure_interval=False,
            ),
        )


register_dataset(DATASET_NAME, SpellBackwardDataset, SpellBackwardConfig, SpellBackwardCurriculum)
