"""
Game tasks for training reasoning capabilities:
- Board games
- Puzzle games
- Strategy games
"""

from .sudoku import SudokuConfig, SudokuDataset, sudoku_dataset

__all__ = ["SudokuConfig", "SudokuDataset", "sudoku_dataset"]
