"""
Game tasks for training reasoning capabilities:
- Board games
- Puzzle games
- Strategy games
- Simulation games
"""

from .countdown import CountdownConfig, CountdownDataset
from .emoji_mystery import EmojiMysteryConfig, EmojiMysteryDataset
from .futoshiki import FutoshikiConfig, FutoshikiDataset
from .knight_swap import KnightSwapConfig, KnightSwapDataset
from .maze import MazeConfig, MazeDataset
from .mini_sudoku import MiniSudokuConfig, MiniSudokuDataset
from .n_queens import NQueensDataset
from .rush_hour import RushHourConfig, RushHourDataset
from .sokoban import SokobanConfig, SokobanDataset
from .sudoku import SudokuConfig, SudokuDataset
from .tower_of_hanoi import HanoiConfig, HanoiDataset
from .tsumego import TsumegoConfig, TsumegoDataset

__all__ = [
    "CountdownConfig",
    "CountdownDataset",
    "EmojiMysteryConfig",
    "EmojiMysteryDataset",
    "FutoshikiConfig",
    "FutoshikiDataset",
    "MiniSudokuConfig",
    "MiniSudokuDataset",
    "SudokuConfig",
    "SudokuDataset",
    "SokobanConfig",
    "SokobanDataset",
    "RushHourConfig",
    "RushHourDataset",
    "MazeConfig",
    "MazeDataset",
    "HanoiConfig",
    "HanoiDataset",
    "NQueensDataset",
    "TsumegoConfig",
    "TsumegoDataset",
    "KnightSwapConfig",
    "KnightSwapDataset",
]
