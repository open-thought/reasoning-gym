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
from .sentence_reordering import SentenceReorderingConfig, SentenceReorderingDataset
from .spell_backward import SpellBackwardConfig, SpellBackwardDataset
from .word_sequence_reversal import WordSequenceReversalConfig, WordSequenceReversalDataset
from .word_sorting import WordSortingConfig, WordSortingDataset, TextTransformation

__all__ = [
    "SpellBackwardConfig",
    "SpellBackwardDataset",
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
    "SentenceReorderingConfig",
    "SentenceReorderingDataset",
    "WordSequenceReversalConfig",
    "WordSequenceReversalDataset",
    "WordSortingConfig",
    "WordSortingDataset",
    "TextTransformation",
]
