"""Spell backward task generator"""

import re
from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from ..data import read_data_file
from ..factory import ProceduralDataset, register_dataset


@dataclass
class SpellBackwardConfig:
    """Configuration for spelling words backward task generation"""

    min_word_len: int = 3  # Minimum word length
    seed: Optional[int] = None
    size: int = 500  # Virtual dataset size

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.min_word_len > 0, "min_word_len must be positive"


class SpellBackwardDataset(ProceduralDataset):
    """Generates tasks to spell words backward"""

    def __init__(self, config: SpellBackwardConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

        # Load and preprocess text
        text = read_data_file("in_the_year_2889.txt")
        # Extract words and clean them to contain only alphanumeric characters
        self.words = [
            word for word in re.findall(r"\b\w+\b", text) if word.isalnum() and len(word) >= config.min_word_len
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
            "metadata": {"word": word, "word_len": len(word)},
        }

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        reward = 0.0
        expected_answer = entry["answer"]
        if answer is not None:
            try:
                if expected_answer.lower() == answer.lower():
                    reward = 1.0
                else:
                    reward = 0.05
            except:
                reward = 0.01
        return reward


register_dataset("spell_backward", SpellBackwardDataset, SpellBackwardConfig)
