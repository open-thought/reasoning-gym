"""Word reversal task generator"""

import re
from dataclasses import dataclass
from random import Random
from typing import List, Optional

from ..data import read_data_file
from ..dataset import ProceduralDataset


@dataclass
class WordReversalConfig:
    """Configuration for word reversal task generation"""

    min_words: int = 3  # Minimum words in list
    max_words: int = 8  # Maximum words in list
    seed: Optional[int] = None
    size: int = 500  # Virtual dataset size

    def validate(self):
        """Validate configuration parameters"""
        assert self.min_words > 0, "min_words must be positive"
        assert self.max_words >= self.min_words, "max_words must be >= min_words"


class WordReversalDataset(ProceduralDataset):
    """Generates word reversal tasks from text spans"""

    def __init__(self, config: WordReversalConfig):
        self.config = config
        self.config.validate()
        super().__init__(seed=config.seed, size=config.size)

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


def word_reversal_dataset(
    min_words: int = 3,
    max_words: int = 8,
    seed: Optional[int] = None,
    size: int = 500,
) -> WordReversalDataset:
    """Create a WordReversalDataset with the given configuration."""
    config = WordReversalConfig(
        min_words=min_words,
        max_words=max_words,
        seed=seed,
        size=size,
    )
    return WordReversalDataset(config)
