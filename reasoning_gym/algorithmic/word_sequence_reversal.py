"""Word reversal task generator"""

import re
from dataclasses import dataclass
from random import Random
from typing import Optional

from ..coaching import AttributeType, BaseCurriculum, RangeAttributeDefinition
from ..data import read_data_file
from ..factory import ProceduralDataset, register_dataset

QUESTION_TEMPLATE = """Solve the following problem.

Provide you answer as a comma-separated list of words with a space after the comma.

Reverse this list of words: {words}
"""


@dataclass
class WordSequenceReversalConfig:
    """Configuration for word sequence reversal task generation"""

    min_words: int = 3  # Minimum words in list
    max_words: int = 8  # Maximum words in list
    seed: Optional[int] = None
    size: int = 500  # Virtual dataset size

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.min_words > 0, "min_words must be positive"
        assert self.max_words >= self.min_words, "max_words must be >= min_words"


class WordSequenceReversalDataset(ProceduralDataset):
    """Generates word sequence reversal tasks from text spans"""

    def __init__(self, config: WordSequenceReversalConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

        # Load and preprocess text
        text = read_data_file("in_the_year_2889.txt")
        # Extract words and clean them to contain only alphanumeric characters
        self.words = [word for word in re.findall(r"\b\w+\b", text) if word.isalnum()]

    def __getitem__(self, idx: int) -> dict:
        """Generate a single word reversal task"""
        rng = Random(self.seed + idx)

        # Select random words
        num_words = min(
            rng.randint(self.config.min_words, self.config.max_words),
            len(self.words),
        )
        word_indices = rng.sample(range(len(self.words)), num_words)
        words = [self.words[i] for i in word_indices]

        # Create question and answer
        words_str = ", ".join(words)
        answer = ", ".join(reversed(words))

        return {
            "question": f"{QUESTION_TEMPLATE.format(words=words_str)}",
            "answer": answer,
            "metadata": {"num_words": num_words, "words": words, "difficulty": {"words": num_words}},
        }


class WordSequenceReversalCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(WordSequenceReversalCurriculum.__name__, WordSequenceReversalConfig)

        # Define attributes
        self._define_attributes(
            RangeAttributeDefinition(
                name="words",
                levels=[10, 50, 100, 500],
                default_level=1,
                description="Number of words in the list",
                attr_type=AttributeType.APPEND,
                min_value=2,
                lower_field_name="min_words",
                upper_field_name="max_words",
            ),
        )


register_dataset(
    "word_sequence_reversal", WordSequenceReversalDataset, WordSequenceReversalConfig, WordSequenceReversalCurriculum
)
