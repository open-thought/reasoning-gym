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
    combinatorics,
    data,
    games,
    geometry,
    graphs,
    induction,
    languages,
    logic,
    optimization,
    probability,
    statistics,
)
from .factory import create_dataset, get_score_answer_fn, register_dataset
from .scoring import cascade_score, float_match, math_match, string_match, strip_latex

__version__ = "0.1.19"
__all__ = [
    "arc",
    "algebra",
    "algorithmic",
    "arithmetic",
    "code",
    "cognition",
    "combinatorics",
    "data",
    "games",
    "geometry",
    "graphs",
    "languages",
    "logic",
    "induction",
    "optimization",
    "probability",
    "statistics",
    "create_dataset",
    "register_dataset",
    "get_score_answer_fn",
    "cascade_score",
    "strip_latex",
    "string_match",
    "float_match",
    "math_match",
]
