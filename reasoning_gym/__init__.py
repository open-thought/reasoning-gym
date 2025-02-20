"""
Reasoning Gym - A library of procedural dataset generators for training reasoning models
"""

from . import algebra, algorithmic, arc, arithmetic, code, cognition, data, games, geometry, graphs, logic
from .factory import create_dataset, register_dataset

__version__ = "0.1.9"
__all__ = [
    "arc",
    "algebra",
    "algorithmic",
    "arithmetic",
    "code",
    "cognition",
    "data",
    "games",
    "geometry",
    "graphs",
    "logic",
    "create_dataset",
    "register_dataset",
]
