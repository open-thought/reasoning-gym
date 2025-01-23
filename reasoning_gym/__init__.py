"""
Reasoning Gym - A library of procedural dataset generators for training reasoning models
"""

from . import arithmetic
from . import algorithmic
from . import cognition
from . import data
from . import logic

__version__ = "0.1.0"
__all__ = ["arithmetic", "algorithmic", "cognition", "data", "logic"]
