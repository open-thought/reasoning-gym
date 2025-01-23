"""
Algorithmic tasks for training reasoning capabilities:
- Text processing
- Counting
- Sorting
- Pattern matching
"""

from .letter_counting import LetterCountingConfig, LetterCountingDataset, letter_counting_dataset
from .word_reversal import WordReversalConfig, WordReversalDataset, word_reversal_dataset

__all__ = [
    "LetterCountingConfig", 
    "LetterCountingDataset", 
    "letter_counting_dataset",
    "WordReversalConfig",
    "WordReversalDataset",
    "word_reversal_dataset"
]
