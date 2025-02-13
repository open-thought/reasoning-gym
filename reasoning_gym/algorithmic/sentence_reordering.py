"""Sentence re-ordering task generator"""

import re
from dataclasses import dataclass
from random import Random
from typing import Optional

from ..data import read_data_file
from ..factory import ProceduralDataset, register_dataset


@dataclass
class SentenceReorderingConfig:
    """Configuration for sentence reordering task generation"""

    min_words_in_sentence: int = 3
    max_words_in_sentence: int = 20
    seed: Optional[int] = None
    size: int = 500

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.min_words_in_sentence > 0, "min_words_in_sentence must be positive"
        assert (
            self.max_words_in_sentence >= self.min_words_in_sentence
        ), "max_words_in_sentence must be >= min_words_in_sentence"
        assert (
            self.max_words_in_sentence >= self.min_words_in_sentence
        ), "max_words_in_sentence must be >= min_words_in_sentence"


class SentenceReorderingDataset(ProceduralDataset):
    """Generates sentence reordering tasks from text spans"""

    def __init__(self, config: SentenceReorderingConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

        # Load and preprocess text
        text = read_data_file("in_the_year_2889.txt")
        # Extract sentences make sure they are greater than or equal to the number of words in a sentence
        # Ensure that only the length of alphanumeric characters in the sentence is considered
        self.sentences = [
            sentence.strip()
            for sentence in re.findall(r"[^.!?]+[.!?]", text)  # Changed pattern to include the ending punctuation
            if self.config.min_words_in_sentence
            <= len(re.findall(r"\b\w+\b", sentence))
            <= self.config.max_words_in_sentence
        ]

    def _generate_sentence_dataset(self, sentence: str, seed: int, idx: int, shuffle=True):
        """
        Generate a procedural dataset by shuffling the words in the input sentence.

        Args:
            sentence (str): The correct sentence to use for dataset generation.
            seed (int): The seed to use for random number generation.
            idx (int): The index to add to the seed for random number generation.
            shuffle (bool): Whether to shuffle the words to create the input sentence.

        Returns:
            dict: A dictionary containing the input sentence and the correct sentence (goal).
        """
        rng = Random(seed + idx)
        words = sentence.split()  # Split the sentence into words
        scrambled_words = words.copy()
        if shuffle:
            rng.shuffle(scrambled_words)  # Shuffle the words to generate the input
        input_sentence = " ".join(scrambled_words)
        goal_sentence = " ".join(words)
        return {"input": input_sentence, "goal": goal_sentence}

    def __getitem__(self, idx: int) -> dict:
        """Generate a single sentence reordering task"""
        rng = Random(self.seed + idx)
        sentence_dataset = self._generate_sentence_dataset(rng.choice(self.sentences), self.seed, idx)

        # Ensure only 'input' and 'goal' keys are present
        if set(sentence_dataset.keys()) != {"input", "goal"}:
            raise KeyError("The dictionary must contain only 'input' and 'goal' keys")

        # Solve the task by sorting words to match the goal sentence
        input_words = sentence_dataset["input"].split()
        question = " ".join(input_words)
        goal_words = sentence_dataset["goal"].split()
        solved_sentence = " ".join(sorted(input_words, key=lambda word: goal_words.index(word)))
        # Check for length of alphanumeric characters in the solved sentence
        word_count = len(re.findall(r"\b\w+\b", solved_sentence))

        return {
            "question": f"Restore the correct order of words in the following sentence: {question}",
            "answer": solved_sentence,
            "metadata": {"word_count": word_count},
        }


register_dataset("sentence_reordering", SentenceReorderingDataset, SentenceReorderingConfig)
