"""
Logic tasks for training reasoning capabilities:
- Propositional logic
- Predicate logic
- Set theory
- Syllogisms
"""

from .propositional_logic import PropositionalLogicConfig, PropositionalLogicDataset, propositional_logic_dataset
from .syllogisms import SyllogismConfig, SyllogismDataset, Term, syllogism_dataset

__all__ = [
    "PropositionalLogicConfig",
    "PropositionalLogicDataset",
    "propositional_logic_dataset",
    "SyllogismConfig",
    "SyllogismDataset",
    "syllogism_dataset",
    "Term",
]
