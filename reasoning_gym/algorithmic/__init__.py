"""
Algorithmic tasks for training reasoning capabilities:
- Text processing
- Counting
- Sorting
- Pattern matching
"""

from reasoning_gym.arithmetic.basic_arithmetic import arithmetic_dataset
from reasoning_gym.arithmetic.chain_sum import chain_sum_dataset
from .base_conversion import BaseConversionConfig, BaseConversionDataset, base_conversion_dataset
from .letter_counting import LetterCountingConfig, LetterCountingDataset, letter_counting_dataset
from .number_filtering import NumberFilteringConfig, NumberFilteringDataset, number_filtering_dataset
from .word_reversal import WordReversalConfig, WordReversalDataset, word_reversal_dataset

__all__ = [
    "arithmetic_dataset",
    "BaseConversionConfig",
    "BaseConversionDataset",
    "base_conversion_dataset",
    "chain_sum_dataset",
    "LetterCountingConfig", 
    "LetterCountingDataset", 
    "letter_counting_dataset",
    "NumberFilteringConfig",
    "NumberFilteringDataset",
    "number_filtering_dataset",
    "WordReversalConfig",
    "WordReversalDataset",
    "word_reversal_dataset"
]
