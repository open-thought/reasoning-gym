"""
Game tasks for training reasoning capabilities:
- Board games
- Puzzle games
- Strategy games
"""

from .countdown import CountdownConfig, CountdownDataset
from .maze import MazeConfig, MazeDataset
from .mini_sudoku import MiniSudokuConfig, MiniSudokuDataset
from .sudoku import SudokuConfig, SudokuDataset

__all__ = [
    "CountdownConfig",
    "CountdownDataset",
    "MiniSudokuConfig",
    "MiniSudokuDataset",
    "SudokuConfig",
    "SudokuDataset",
    "MazeConfig",
    "MazeDataset",
]
