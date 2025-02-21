"""Word letter jumbling task generator"""

import re
from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from reasoning_gym.data import read_data_file

from ..factory import ProceduralDataset, register_dataset

QUESTION_TEMPLATE = """Your task is to unsramble words in a sentence.

For each word in a sentence, the letter may have been randomly shuffled. Your task is to unscramble the words.

The order of the words in the sentence is preserved. Moreover, the style of the sentence is preserved (i.e. punctuation, capitalization, new lines, etc.).

Example:
- Input: Unscramble these words: raendgmeins yWh nya hilcd anc od hatt
- Output: meanderings Why any child can do that
- Explanation
    - We unscramble each of the words independently.
    - raendgmeins -> meanderings
    - yWh -> Why
    - nya -> any
    - hilcd -> child
    - anc -> can
    - od -> do
    - hatt -> that
    - The final answer is: meanderings Why any child can do that
    - Notice that the order of the words is preserved, no new words / symbols (e.g. new lines) are added.

Now, unscramble these words: {words}
"""


@dataclass
class LetterJumbleConfig:
    """Configuration for letter jumbling task generation"""

    min_word_len: int = 1  # Minimum word length
    max_word_len: int = 64  # Maximum word length
    min_words: int = 3  # Minimum words per task
    max_words: int = 20  # Maximum words per task
    min_corruption_level: float = 0.1  # Minimum fraction of characters to swap
    max_corruption_level: float = 0.9  # Maximum fraction of characters to swap
    consecutive_words: bool = True  # Whether to select consecutive words from text
    seed: Optional[int] = None
    size: int = 500  # Virtual dataset size

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.min_word_len > 0, "min_word_len must be positive"
        assert self.max_word_len >= self.min_word_len, "max_word_len must be >= min_word_len"
        assert self.min_words > 0, "min_words must be positive"
        assert self.max_words >= self.min_words, "max_words must be >= min_words"
        assert 0 <= self.min_corruption_level <= 1, "min_corruption_level must be in [0,1]"
        assert 0 <= self.max_corruption_level <= 1, "max_corruption_level must be in [0,1]"
        assert (
            self.max_corruption_level >= self.min_corruption_level
        ), "max_corruption_level must be >= min_corruption_level"


class LetterJumbleDataset(ProceduralDataset):
    """Generates word letter jumbling tasks"""

    def __init__(self, config: LetterJumbleConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

        # Load and preprocess text
        text = read_data_file("in_the_year_2889.txt")
        # Extract words and filter by length
        self.words = [
            word
            for word in re.findall(r"\b\w+\b", text)
            if self.config.min_word_len <= len(word) <= self.config.max_word_len and word.isalpha()
        ]

    def _scramble_word(self, word: str, corruption_level: float, rng: Random) -> str:
        """Scramble a word by swapping random pairs of characters"""
        if len(word) < 2:  # Can't scramble 1-character words
            return word

        word = list(word)
        num_swaps = max(1, int(len(word) * corruption_level))  # Ensure at least one swap

        for _ in range(num_swaps):
            # Pick two different random positions
            pos1, pos2 = rng.sample(range(len(word)), 2)
            # Swap characters
            word[pos1], word[pos2] = word[pos2], word[pos1]

        return "".join(word)

    def __getitem__(self, idx: int) -> dict:
        """Generate a single word jumbling task"""
        rng = Random(self.seed + idx)

        # Select number of words and corruption level
        num_words = rng.randint(self.config.min_words, self.config.max_words)
        corruption_level = rng.uniform(self.config.min_corruption_level, self.config.max_corruption_level)

        # Select words based on configuration
        if self.config.consecutive_words:
            # Select consecutive words from a random starting position
            start_idx = rng.randint(0, len(self.words) - num_words)
            selected_words = self.words[start_idx : start_idx + num_words]
        else:
            # Select random words
            selected_words = rng.sample(self.words, num_words)

        # Scramble each word
        scrambled_words = [self._scramble_word(word, corruption_level, rng) for word in selected_words]

        return {
            "question": QUESTION_TEMPLATE.format(words=" ".join(scrambled_words)),
            "answer": " ".join(selected_words),
            "metadata": {
                "num_words": num_words,
                "corruption_level": corruption_level,
                "scrambled_words": scrambled_words,
                "original_words": selected_words,
            },
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

        oracle_answer = entry["answer"].strip()
        if answer:
            answer = answer.strip()
            if answer == oracle_answer:
                return 1.0
            elif answer.lower() == oracle_answer.lower():
                return 0.5
            else:
                return 0.01
        return 0.0


register_dataset("letter_jumble", LetterJumbleDataset, LetterJumbleConfig)
