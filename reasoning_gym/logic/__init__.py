"""
Logic tasks for training reasoning capabilities.
"""

from .aiw import AliceInWonderlandConfig, AliceInWonderlandDataset
from .propositional_logic import PropositionalLogicConfig, PropositionalLogicDataset
from .self_reference import SelfReferenceConfig, SelfReferenceDataset
from .syllogisms import SyllogismConfig, SyllogismDataset, Term
from .zebra_puzzles import ZebraConfig, ZebraDataset
from .circuit_logic import CircuitLogicConfig, CircuitLogicDataset

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
    "SelfReference",
    "SelfReferenceDataset",
    "CircuitLogicConfig",
    "CircuitLogicDataset",
]
