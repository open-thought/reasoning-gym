"""
Game tasks for training reasoning capabilities:
- Board games
- Puzzle games
- Strategy games
"""

from .maze import MazeConfig, MazeDataset
from .mini_sudoku import MiniSudokuConfig, MiniSudokuDataset
from .sudoku import SudokuConfig, SudokuDataset

__all__ = [
    "MiniSudokuConfig",
    "MiniSudokuDataset",
    "SudokuConfig",
    "SudokuDataset",
    "MazeConfig",
    "MazeDataset",
]
