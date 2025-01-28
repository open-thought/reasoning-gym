"""
Reasoning Gym - A library of procedural dataset generators for training reasoning models
"""

from . import algebra, algorithmic, arithmetic, cognition, data, games, graphs, logic
from .factory import create_dataset, register_dataset

__version__ = "0.1.1"
__all__ = [
    "arithmetic",
    "algorithmic",
    "algebra",
    "cognition",
    "data",
    "games",
    "graphs",
    "logic",
    "create_dataset",
    "register_dataset",
]
