"""
Algorithmic tasks for training reasoning capabilities:
- Text processing
- Counting
- Sorting
- Pattern matching
"""

from .ab import ABConfig, ABDataset
from .base_conversion import BaseConversionConfig, BaseConversionDataset
from .binary_alternation import BinaryAlternationConfig, BinaryAlternationDataset
from .binary_matrix import BinaryMatrixConfig, BinaryMatrixDataset
from .caesar_cipher import CaesarCipherConfig, CaesarCipherDataset
from .count_primes import CountPrimesConfig, CountPrimesDataset
from .cryptarithm import CryptarithmConfig, CryptarithmDataset
from .game_of_life import GameOfLifeConfig, GameOfLifeDataset
from .graph_color import GraphColorConfig, GraphColorDataset
from .group_anagrams import GroupAnagramsConfig, GroupAnagramsDataset
from .isomorphic_strings import IsomorphicStringsConfig, IsomorphicStringsDataset
from .jugs import JugsConfig, JugsDataset
from .letter_counting import LetterCountingConfig, LetterCountingDataset
from .letter_jumble import LetterJumbleConfig, LetterJumbleDataset
from .manipulate_matrix import ManipulateMatrixConfig, ManipulateMatrixDataset
from .number_filtering import NumberFilteringConfig, NumberFilteringDataset
from .number_sorting import NumberSortingConfig, NumberSortingDataset
from .palindrome_generation import PalindromeConfig, PalindromeDataset
from .palindrome_partitioning import PalindromePartitioningConfig, PalindromePartitioningDataset
from .pool_matrix import PoolMatrixConfig, PoolMatrixDataset
from .ransom_note import RansomNoteConfig, RansomNoteDataset
from .rotate_matrix import RotateMatrixConfig, RotateMatrixDataset
from .rotten_oranges import RottenOrangesConfig, RottenOrangesDataset
from .sentence_reordering import SentenceReorderingConfig, SentenceReorderingDataset
from .spell_backward import SpellBackwardConfig, SpellBackwardDataset
from .spiral_matrix import SpiralMatrixConfig, SpiralMatrixDataset
from .string_insertion import StringInsertionConfig, StringInsertionDataset
from .string_manipulation import StringManipulationConfig, StringManipulationDataset
from .string_splitting import StringSplittingConfig, StringSplittingDataset
from .string_synthesis import StringSynthesisConfig, StringSynthesisDataset
from .word_ladder import WordLadderConfig, WordLadderDataset
from .word_sequence_reversal import WordSequenceReversalConfig, WordSequenceReversalDataset
from .word_sorting import TextTransformation, WordSortingConfig, WordSortingDataset

__all__ = [
    "SpellBackwardConfig",
    "SpellBackwardDataset",
    "BaseConversionConfig",
    "BaseConversionDataset",
    "CaesarCipherConfig",
    "CaesarCipherDataset",
    "CryptarithmConfig",
    "CryptarithmDataset",
    "GameOfLifeConfig",
    "GameOfLifeDataset",
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
    "WordLadderConfig",
    "WordLadderDataset",
    "PalindromeConfig",
    "PalindromeDataset",
    "GroupAnagramsConfig",
    "GroupAnagramsDataset",
    "PalindromePartitioningConfig",
    "PalindromePartitioningDataset",
    "SpiralMatrixConfig",
    "SpiralMatrixDataset",
    "RansomNoteConfig",
    "RansomNoteDataset",
    "IsomorphicStringsConfig",
    "IsomorphicStringsDataset",
    "RotateMatrixConfig",
    "RotateMatrixDataset",
    "ManipulateMatrixConfig",
    "ManipulateMatrixDataset",
    "BinaryMatrixConfig",
    "BinaryMatrixDataset",
    "PoolMatrixConfig",
    "PoolMatrixDataset",
    "ABConfig",
    "ABDataset",
    "CountPrimesConfig",
    "CountPrimesDataset",
    "GraphColorConfig",
    "GraphColorDataset",
    "StringInsertionConfig",
    "StringInsertionDataset",
    "StringManipulationConfig",
    "StringManipulationDataset",
    "StringSplittingConfig",
    "StringSplittingDataset",
    "StringSynthesisConfig",
    "StringSynthesisDataset",
    "RottenOrangesConfig",
    "RottenOrangesDataset",
    "JugsConfig",
    "JugsDataset",
    "BinaryAlternationConfig",
    "BinaryAlternationDataset",
]
