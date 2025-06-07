"""
Game tasks for training reasoning capabilities:
- Board games
- Puzzle games
- Strategy games
- Simulation games
"""

from .boxnet import BoxnetConfig, BoxnetCurriculum, BoxnetDataset
from .countdown import CountdownConfig, CountdownCurriculum, CountdownDataset
from .emoji_mystery import EmojiMysteryConfig, EmojiMysteryCurriculum, EmojiMysteryDataset
from .futoshiki import FutoshikiConfig, FutoshikiCurriculum, FutoshikiDataset
from .kakurasu import KakurasuConfig, KakurasuCurriculum, KakurasuDataset
from .knight_swap import KnightSwapConfig, KnightSwapCurriculum, KnightSwapDataset
from .mahjong import MahjongPuzzleConfig, MahjongPuzzleCurriculum, MahjongPuzzleDataset
from .maze import MazeConfig, MazeCurriculum, MazeDataset
from .mini_sudoku import MiniSudokuConfig, MiniSudokuCurriculum, MiniSudokuDataset
from .n_queens import NQueensConfig, NQueensCurriculum, NQueensDataset
from .puzzle24 import Puzzle24Config, Puzzle24Curriculum, Puzzle24Dataset
from .rush_hour import RushHourConfig, RushHourCurriculum, RushHourDataset
from .sokoban import SokobanConfig, SokobanCurriculum, SokobanDataset
from .sudoku import SudokuConfig, SudokuCurriculum, SudokuDataset
from .tower_of_hanoi import HanoiConfig, HanoiCurriculum, HanoiDataset
from .tsumego import TsumegoConfig, TsumegoCurriculum, TsumegoDataset

__all__ = [
    "BoxnetConfig",
    "BoxnetDataset",
    "BoxnetCurriculum",
    "CountdownConfig",
    "CountdownDataset",
    "CountdownCurriculum",
    "EmojiMysteryConfig",
    "EmojiMysteryCurriculum",
    "EmojiMysteryDataset",
    "FutoshikiConfig",
    "FutoshikiCurriculum",
    "FutoshikiDataset",
    "KakurasuConfig",
    "KakurasuCurriculum",
    "KakurasuDataset",
    "MiniSudokuConfig",
    "MiniSudokuDataset",
    "MiniSudokuCurriculum",
    "Puzzle24Config",
    "Puzzle24Dataset",
    "Puzzle24Curriculum",
    "SudokuConfig",
    "SudokuCurriculum",
    "SudokuDataset",
    "SokobanConfig",
    "SokobanCurriculum",
    "SokobanDataset",
    "RushHourConfig",
    "RushHourCurriculum",
    "RushHourDataset",
    "MazeConfig",
    "MazeDataset",
    "MazeCurriculum",
    "HanoiConfig",
    "HanoiDataset",
    "HanoiCurriculum",
    "NQueensDataset",
    "NQueensConfig",
    "NQueensCurriculum",
    "TsumegoConfig",
    "TsumegoCurriculum",
    "TsumegoDataset",
    "KnightSwapConfig",
    "KnightSwapDataset",
    "KnightSwapCurriculum",
    "MahjongPuzzleConfig",
    "MahjongPuzzleDataset",
    "MahjongPuzzleCurriculum",
]
