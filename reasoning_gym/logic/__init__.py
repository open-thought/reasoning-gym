"""
Logic tasks for training reasoning capabilities.
"""

from .aiw import AliceInWonderlandConfig, AliceInWonderlandDataset
from .circuit_logic import CircuitLogicConfig, CircuitLogicDataset
from .propositional_logic import PropositionalLogicConfig, PropositionalLogicDataset
from .self_reference import SelfReferenceConfig, SelfReferenceDataset
from .syllogisms import SyllogismConfig, SyllogismDataset
from .zebra_puzzles import ZebraConfig, ZebraDataset

__all__ = [
    "AliceInWonderlandConfig",
    "AliceInWonderlandDataset",
    "PropositionalLogicConfig",
    "PropositionalLogicDataset",
    "SyllogismConfig",
    "SyllogismDataset",
    "syllogism_dataset",
    "ZebraConfig",
    "ZebraDataset",
    "SelfReference",
    "SelfReferenceConfig",
    "SelfReferenceDataset",
    "CircuitLogicConfig",
    "CircuitLogicDataset",
]
