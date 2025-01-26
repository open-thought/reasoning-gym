"""Word reversal task generator"""

import re
from dataclasses import dataclass
from random import Random
from typing import List, Optional

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


@dataclass
class WordReversalConfig:
    """Configuration for word reversal task generation"""

    min_words: int = 3  # Minimum words in list
    max_words: int = 8  # Maximum words in list
    seed: Optional[int] = None
    size: int = 500  # Virtual dataset size

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.min_words > 0, "min_words must be positive"
        assert self.max_words >= self.min_words, "max_words must be >= min_words"


class WordReversalDataset(ProceduralDataset):
    """Generates word reversal tasks from text spans"""

    def __init__(self, config: WordReversalConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

        # Load and preprocess text
        text = read_data_file("in_the_year_2889.txt")
        # Extract words and clean them to contain only alphanumeric characters
        self.words = [word for word in re.findall(r"\b\w+\b", text) if word.isalnum()]

    def __getitem__(self, idx: int) -> dict:
        """Generate a single word reversal task"""
        rng = Random(self.seed + idx)

        # Select random words
        num_words = rng.randint(self.config.min_words, self.config.max_words)
        word_indices = rng.sample(range(len(self.words)), num_words)
        words = [self.words[i] for i in word_indices]

        # Create question and answer
        question = ", ".join(words)
        answer = ", ".join(reversed(words))

        return {
            "question": f"Reverse this list of words: {question}",
            "answer": answer,
            "metadata": {"num_words": num_words, "words": words},
        }


class SpellBackwardDataset(ProceduralDataset):
    """Generates tasks to spell words backward"""

    def __init__(self, config: SpellBackwardConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

        # Load and preprocess text
        text = read_data_file("in_the_year_2889.txt")
        # Extract words and clean them to contain only alphanumeric characters
        self.words = [word for word in re.findall(r"\b\w+\b", text) 
                     if word.isalnum() and len(word) >= config.min_word_len]

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


register_dataset("spell_backward", SpellBackwardDataset, SpellBackwardConfig)
register_dataset("word_reversal", WordReversalDataset, WordReversalConfig)
