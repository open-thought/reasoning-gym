"""
Algorithmic tasks for training reasoning capabilities:
- Text processing
- Counting
- Sorting
- Pattern matching
"""

from .base_conversion import BaseConversionConfig, BaseConversionDataset
from .caesar_cipher import CaesarCipherConfig, CaesarCipherDataset
from .letter_counting import LetterCountingConfig, LetterCountingDataset
from .letter_jumble import LetterJumbleConfig, LetterJumbleDataset
from .number_filtering import NumberFilteringConfig, NumberFilteringDataset
from .number_sorting import NumberSortingConfig, NumberSortingDataset
from .word_reversal import WordReversalConfig, WordReversalDataset
from .sentence_reordering import SentenceReorderingConfig, SentenceReorderingDataset

__all__ = [
    "BaseConversionConfig",
    "BaseConversionDataset",
    "CaesarCipherConfig",
    "CaesarCipherDataset",
    "LetterCountingConfig",
    "LetterCountingDataset",
    "LetterJumbleConfig",
    "LetterJumbleDataset",
    "NumberFilteringConfig",
    "NumberFilteringDataset",
    "NumberSortingConfig",
    "NumberSortingDataset",
    "WordReversalConfig",
    "WordReversalDataset",
    "SentenceReorderingConfig",
    "SentenceReorderingDataset",
]
