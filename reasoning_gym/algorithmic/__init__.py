"""
Algorithmic tasks for training reasoning capabilities:
- Text processing
- Counting
- Sorting
- Pattern matching
"""

from .ab import ABConfig, ABDataset
from .base_conversion import BaseConversionConfig, BaseConversionCurriculum, BaseConversionDataset
from .binary_alternation import BinaryAlternationConfig, BinaryAlternationCurriculum, BinaryAlternationDataset
from .binary_matrix import BinaryMatrixConfig, BinaryMatrixCurriculum, BinaryMatrixDataset
from .caesar_cipher import CaesarCipherConfig, CaesarCipherDataset
from .count_primes import CountPrimesConfig, CountPrimesCurriculum, CountPrimesDataset
from .cryptarithm import CryptarithmConfig, CryptarithmDataset
from .game_of_life import GameOfLifeConfig, GameOfLifeDataset
from .game_of_life_halting import GameOfLifeHaltingConfig, GameOfLifeHaltingDataset
from .graph_color import GraphColorConfig, GraphColorDataset
from .group_anagrams import GroupAnagramsConfig, GroupAnagramsCurriculum, GroupAnagramsDataset
from .isomorphic_strings import IsomorphicStringsConfig, IsomorphicStringsCurriculum, IsomorphicStringsDataset
from .jugs import JugsConfig, JugsDataset
from .letter_counting import LetterCountingConfig, LetterCountingDataset
from .letter_jumble import LetterJumbleConfig, LetterJumbleDataset
from .manipulate_matrix import ManipulateMatrixConfig, ManipulateMatrixCurriculum, ManipulateMatrixDataset
from .number_filtering import NumberFilteringConfig, NumberFilteringDataset
from .number_sorting import NumberSortingConfig, NumberSortingDataset
from .palindrome_generation import PalindromeConfig, PalindromeDataset
from .palindrome_partitioning import (
    PalindromePartitioningConfig,
    PalindromePartitioningCurriculum,
    PalindromePartitioningDataset,
)
from .pool_matrix import PoolMatrixConfig, PoolMatrixDataset
from .ransom_note import RansomNoteConfig, RansomNoteDataset
from .rotate_matrix import RotateMatrixConfig, RotateMatrixCurriculum, RotateMatrixDataset
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
from .word_sorting import TextTransformation, WordSortingConfig, WordSortingCurriculum, WordSortingDataset

__all__ = [
    "SpellBackwardConfig",
    "SpellBackwardDataset",
    "BaseConversionConfig",
    "BaseConversionDataset",
    "BaseConversionCurriculum",
    "CaesarCipherConfig",
    "CaesarCipherDataset",
    "CryptarithmConfig",
    "CryptarithmDataset",
    "GameOfLifeConfig",
    "GameOfLifeDataset",
    "GameOfLifeHaltingConfig",
    "GameOfLifeHaltingDataset",
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
    "WordSortingCurriculum",
    "WordSortingConfig",
    "WordSortingDataset",
    "TextTransformation",
    "WordLadderConfig",
    "WordLadderDataset",
    "PalindromeConfig",
    "PalindromeDataset",
    "GroupAnagramsConfig",
    "GroupAnagramsDataset",
    "GroupAnagramsCurriculum",
    "PalindromePartitioningConfig",
    "PalindromePartitioningDataset",
    "PalindromePartitioningCurriculum",
    "SpiralMatrixConfig",
    "SpiralMatrixDataset",
    "RansomNoteConfig",
    "RansomNoteDataset",
    "IsomorphicStringsConfig",
    "IsomorphicStringsDataset",
    "IsomorphicStringsCurriculum",
    "RotateMatrixConfig",
    "RotateMatrixDataset",
    "RotateMatrixCurriculum",
    "ManipulateMatrixConfig",
    "ManipulateMatrixDataset",
    "ManipulateMatrixCurriculum",
    "BinaryMatrixConfig",
    "BinaryMatrixDataset",
    "BinaryMatrixCurriculum",
    "PoolMatrixConfig",
    "PoolMatrixDataset",
    "ABConfig",
    "ABDataset",
    "CountPrimesConfig",
    "CountPrimesDataset",
    "CountPrimesCurriculum",
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
    "BinaryAlternationCurriculum",
]
