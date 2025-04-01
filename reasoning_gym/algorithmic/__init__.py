"""
Algorithmic tasks for training reasoning capabilities:
- Text processing
- Counting
- Sorting
- Pattern matching
"""

from .ab import ABConfig, ABCurriculum, ABDataset
from .base_conversion import BaseConversionConfig, BaseConversionCurriculum, BaseConversionDataset
from .binary_alternation import BinaryAlternationConfig, BinaryAlternationCurriculum, BinaryAlternationDataset
from .binary_matrix import BinaryMatrixConfig, BinaryMatrixCurriculum, BinaryMatrixDataset
from .caesar_cipher import CaesarCipherConfig, CaesarCipherCurriculum, CaesarCipherDataset
from .count_primes import CountPrimesConfig, CountPrimesCurriculum, CountPrimesDataset
from .cryptarithm import CryptarithmConfig, CryptarithmCurriculum, CryptarithmDataset
from .game_of_life import GameOfLifeConfig, GameOfLifeCurriculum, GameOfLifeDataset
from .game_of_life_halting import GameOfLifeHaltingConfig, GameOfLifeHaltingDataset
from .graph_color import GraphColorConfig, GraphColorCurriculum, GraphColorDataset
from .group_anagrams import GroupAnagramsConfig, GroupAnagramsCurriculum, GroupAnagramsDataset
from .isomorphic_strings import IsomorphicStringsConfig, IsomorphicStringsCurriculum, IsomorphicStringsDataset
from .jugs import JugsConfig, JugsCurriculum, JugsDataset
from .letter_counting import LetterCountingConfig, LetterCountingCurriculum, LetterCountingDataset
from .letter_jumble import LetterJumbleConfig, LetterJumbleCurriculum, LetterJumbleDataset
from .manipulate_matrix import ManipulateMatrixConfig, ManipulateMatrixCurriculum, ManipulateMatrixDataset
from .number_filtering import NumberFilteringConfig, NumberFilteringCurriculum, NumberFilteringDataset
from .number_sorting import NumberSortingConfig, NumberSortingCurriculum, NumberSortingDataset
from .palindrome_generation import PalindromeConfig, PalindromeCurriculum, PalindromeDataset
from .palindrome_partitioning import (
    PalindromePartitioningConfig,
    PalindromePartitioningCurriculum,
    PalindromePartitioningDataset,
)
from .pool_matrix import PoolMatrixConfig, PoolMatrixCurriculum, PoolMatrixDataset
from .ransom_note import RansomNoteConfig, RansomNoteCurriculum, RansomNoteDataset
from .rotate_matrix import RotateMatrixConfig, RotateMatrixCurriculum, RotateMatrixDataset
from .rotten_oranges import RottenOrangesConfig, RottenOrangesCurriculum, RottenOrangesDataset
from .sentence_reordering import SentenceReorderingConfig, SentenceReorderingCurriculum, SentenceReorderingDataset
from .spell_backward import SpellBackwardConfig, SpellBackwardCurriculum, SpellBackwardDataset
from .spiral_matrix import SpiralMatrixConfig, SpiralMatrixCurriculum, SpiralMatrixDataset
from .string_insertion import StringInsertionConfig, StringInsertionCurriculum, StringInsertionDataset
from .string_manipulation import StringManipulationConfig, StringManipulationDataset
from .string_splitting import StringSplittingConfig, StringSplittingCurriculum, StringSplittingDataset
from .string_synthesis import StringSynthesisConfig, StringSynthesisCurriculum, StringSynthesisDataset
from .word_ladder import WordLadderConfig, WordLadderCurriculum, WordLadderDataset
from .word_sequence_reversal import (
    WordSequenceReversalConfig,
    WordSequenceReversalCurriculum,
    WordSequenceReversalDataset,
)
from .word_sorting import TextTransformation, WordSortingConfig, WordSortingCurriculum, WordSortingDataset

__all__ = [
    "SpellBackwardConfig",
    "SpellBackwardDataset",
    "SpellBackwardCurriculum",
    "BaseConversionConfig",
    "BaseConversionDataset",
    "BaseConversionCurriculum",
    "CaesarCipherConfig",
    "CaesarCipherDataset",
    "CaesarCipherCurriculum",
    "CryptarithmConfig",
    "CryptarithmDataset",
    "CryptarithmCurriculum",
    "GameOfLifeConfig",
    "GameOfLifeDataset",
    "GameOfLifeCurriculum",
    "GameOfLifeHaltingConfig",
    "GameOfLifeHaltingDataset",
    "LetterCountingConfig",
    "LetterCountingDataset",
    "LetterCountingCurriculum",
    "LetterJumbleConfig",
    "LetterJumbleDataset",
    "LetterJumbleCurriculum",
    "NumberFilteringConfig",
    "NumberFilteringDataset",
    "NumberFilteringCurriculum",
    "NumberSortingConfig",
    "NumberSortingDataset",
    "NumberSortingCurriculum",
    "SentenceReorderingConfig",
    "SentenceReorderingDataset",
    "SentenceReorderingCurriculum",
    "WordSequenceReversalConfig",
    "WordSequenceReversalDataset",
    "WordSequenceReversalCurriculum",
    "WordSortingCurriculum",
    "WordSortingConfig",
    "WordSortingDataset",
    "TextTransformation",
    "WordLadderConfig",
    "WordLadderCurriculum",
    "WordLadderDataset",
    "PalindromeConfig",
    "PalindromeDataset",
    "PalindromeCurriculum",
    "GroupAnagramsConfig",
    "GroupAnagramsDataset",
    "GroupAnagramsCurriculum",
    "PalindromePartitioningConfig",
    "PalindromePartitioningDataset",
    "PalindromePartitioningCurriculum",
    "SpiralMatrixConfig",
    "SpiralMatrixDataset",
    "SpiralMatrixCurriculum",
    "RansomNoteConfig",
    "RansomNoteDataset",
    "RansomNoteCurriculum",
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
    "PoolMatrixCurriculum",
    "ABConfig",
    "ABDataset",
    "ABCurriculum",
    "CountPrimesConfig",
    "CountPrimesDataset",
    "CountPrimesCurriculum",
    "GraphColorConfig",
    "GraphColorDataset",
    "GraphColorCurriculum",
    "StringInsertionConfig",
    "StringInsertionDataset",
    "StringInsertionCurriculum",
    "StringManipulationConfig",
    "StringManipulationDataset",
    "StringManipulationCurriculum",
    "StringSplittingConfig",
    "StringSplittingDataset",
    "StringSplittingCurriculum",
    "StringSynthesisConfig",
    "StringSynthesisDataset",
    "StringSynthesisCurriculum",
    "RottenOrangesConfig",
    "RottenOrangesDataset",
    "RottenOrangesCurriculum",
    "JugsConfig",
    "JugsDataset",
    "JugsCurriculum",
    "BinaryAlternationConfig",
    "BinaryAlternationDataset",
    "BinaryAlternationCurriculum",
]
