"""
Logic tasks for training reasoning capabilities:
- Propositional logic
- Predicate logic
- Set theory
- Syllogisms
"""

from .propositional_logic import PropositionalLogicConfig, PropositionalLogicDataset
from .syllogisms import SyllogismConfig, SyllogismDataset, Term

__all__ = [
    "PropositionalLogicConfig",
    "PropositionalLogicDataset",
    "SyllogismConfig",
    "SyllogismDataset",
    "syllogism_dataset",
    "Term",
]
