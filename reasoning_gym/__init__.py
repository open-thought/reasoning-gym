"""
Reasoning Gym - A library of procedural dataset generators for training reasoning models
"""

from . import (
    algebra,
    algorithmic,
    arc,
    arithmetic,
    code,
    cognition,
    data,
    games,
    geometry,
    graphs,
    induction,
    logic,
    probability,
)
from .factory import create_dataset, get_score_answer_fn, register_dataset

__version__ = "0.1.19"
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
    "induction",
    "probability",
    "create_dataset",
    "register_dataset",
    "get_score_answer_fn",
]
