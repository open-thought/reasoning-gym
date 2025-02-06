"""
Logic tasks for training reasoning capabilities.
"""

from .aiw import AliceInWonderlandConfig, AliceInWonderlandDataset
from .propositional_logic import PropositionalLogicConfig, PropositionalLogicDataset
from .syllogisms import SyllogismConfig, SyllogismDataset, Term
from .zebra_puzzles import ZebraConfig, ZebraDataset

__all__ = [
    "AliceInWonderlandConfig",
    "AliceInWonderlandDataset",
    "PropositionalLogicConfig",
    "PropositionalLogicDataset",
    "SyllogismConfig",
    "SyllogismDataset",
    "syllogism_dataset",
    "Term",
    "ZebraConfig",
    "ZebraDataset",
]
