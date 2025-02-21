"""Word sorting task generator"""

import re
from dataclasses import dataclass
from enum import StrEnum
from random import Random
from typing import Any, Optional

from ..data import read_data_file
from ..factory import ProceduralDataset, register_dataset


class TextTransformation(StrEnum):
    """Text transformation options"""

    LOWERCASE = "lowercase"
    UPPERCASE = "uppercase"
    ORIGINAL = "original"
    RANDOMCASE = "randomcase"


QUESTION_TEMPLATE = """Your task is to sort words in ascending or descending order using ASCII/Unicode ordering.

Example:
- Input: Sort these words in ascending order (using ASCII/Unicode ordering) and return them as a comma-separated list: freely, idea, indemnify, last, END, solving
- Output: END, freely, idea, indemnify, last, solving
- Explanation:
    - Uppercase letters come before lowercase letters, hence why "END" comes first.
    - "freely" comes before "idea" because "f" comes before "i".
    - "idea" comes before "indemnify" because even though they both start with "i", "d" comes before "n".
    - "indemnify" comes before "last" because "i" comes before "l".
    - "last" comes before "solving" because "l" comes before "s".
    - Finally, the output is provided as a comma separated list of the sorted words.

Now, sort these words in {direction} order (using ASCII/Unicode ordering) and return them as a comma-separated list: {words}
"""


@dataclass
class WordSortingConfig:
    """Configuration for word sorting task generation"""

    min_words: int = 3  # Minimum words to sort
    max_words: int = 10  # Maximum words to sort
    min_word_length: int = 3  # Minimum word length
    max_word_length: int = 12  # Maximum word length
    transformation: TextTransformation = TextTransformation.ORIGINAL
    seed: Optional[int] = None
    size: int = 500  # Virtual dataset size

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.min_words > 0, "min_words must be positive"
        assert self.min_words <= self.max_words, "max_words must be >= min_words"
        assert self.min_word_length > 0, "min_word_length must be positive"
        assert self.min_word_length <= self.max_word_length, "max_word_length must be >= min_word_length"
        assert isinstance(self.transformation, TextTransformation), "transformation must be a TextTransformation"


class WordSortingDataset(ProceduralDataset):
    """Generates word sorting tasks"""

    def __init__(self, config: WordSortingConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

        # Load and preprocess text
        text = read_data_file("in_the_year_2889.txt")
        # Extract unique words within length constraints
        self.words = sorted(
            set(
                word
                for word in re.findall(r"\b\w+\b", text)
                if self.config.min_word_length <= len(word) <= self.config.max_word_length
            )
        )

    def _transform_word(self, word: str, rng: Random) -> str:
        """Apply configured transformation to word"""
        if self.config.transformation == TextTransformation.LOWERCASE:
            return word.lower()
        elif self.config.transformation == TextTransformation.UPPERCASE:
            return word.upper()
        elif self.config.transformation == TextTransformation.RANDOMCASE:
            return "".join(c.upper() if rng.choice([True, False]) else c.lower() for c in word)
        return word  # ORIGINAL case

    def _generate_words(self, rng: Random) -> tuple[list[str], list[str]]:
        """Generate list of words and their transformed versions"""
        count = rng.randint(self.config.min_words, self.config.max_words)

        # Select random words
        selected_words = rng.sample(self.words, count)
        # Apply transformation
        transformed_words = [self._transform_word(word, rng) for word in selected_words]

        return selected_words, transformed_words

    def __getitem__(self, idx: int) -> dict:
        """Generate a single sorting task"""
        rng = Random(self.seed + idx)

        original_words, transformed_words = self._generate_words(rng)

        # Generate both ascending and descending answers
        asc_words = sorted(transformed_words)
        desc_words = sorted(transformed_words, reverse=True)

        # Randomly choose ascending or descending
        is_ascending = rng.choice([True, False])
        direction = "ascending" if is_ascending else "descending"
        answer = asc_words if is_ascending else desc_words

        return {
            "question": QUESTION_TEMPLATE.format(direction=direction, words=", ".join(transformed_words)),
            "answer": ", ".join(answer),
            "metadata": {
                "original_words": original_words,
                "transformed_words": transformed_words,
                "direction": direction,
                "transformation": self.config.transformation,
                "sorted_words": answer,
            },
        }

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        oracle_answer = entry["metadata"]["sorted_words"]
        if answer is not None and len(answer) > 0:
            parsed_answer = [word.strip() for word in re.split(r",\s*", answer)]
            if parsed_answer == oracle_answer:
                return 1.0
            elif sorted(parsed_answer) == oracle_answer:
                return 0.2
            else:
                return 0.01

        return 0.0


register_dataset("word_sorting", WordSortingDataset, WordSortingConfig)
