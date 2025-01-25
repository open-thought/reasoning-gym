"""
Game tasks for training reasoning capabilities:
- Board games
- Puzzle games
- Strategy games
"""

from .countdown_game import CountdownGameConfig, CountdownGameDataset
from .maze import MazeConfig, MazeDataset
from .mini_sudoku import MiniSudokuConfig, MiniSudokuDataset
from .sudoku import SudokuConfig, SudokuDataset

__all__ = [
    "CountdownGameConfig",
    "CountdownGameDataset",
    "MiniSudokuConfig",
    "MiniSudokuDataset",
    "SudokuConfig", 
    "SudokuDataset",
    "MazeConfig",
    "MazeDataset",
]
