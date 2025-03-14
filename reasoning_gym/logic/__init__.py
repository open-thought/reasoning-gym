"""
Logic tasks for training reasoning capabilities.
"""

from .aiw import AliceInWonderlandConfig, AliceInWonderlandCurriculum, AliceInWonderlandDataset
from .circuit_logic import CircuitLogicConfig, CircuitLogicDataset
from .knights_knaves import KnightsKnavesConfig, KnightsKnavesDataset
from .propositional_logic import PropositionalLogicConfig, PropositionalLogicDataset, PropositionalLogicCurriculum
from .self_reference import SelfReferenceConfig, SelfReferenceCurriculum, SelfReferenceDataset
from .syllogisms import SyllogismConfig, SyllogismDataset
from .zebra_puzzles import ZebraConfig, ZebraCurriculum, ZebraDataset

__all__ = [
    "AliceInWonderlandConfig",
    "AliceInWonderlandCurriculum",
    "AliceInWonderlandDataset",
    "PropositionalLogicConfig",
    "PropositionalLogicDataset",
    "PropositionalLogicCurriculum",
    "SyllogismConfig",
    "SyllogismDataset",
    "syllogism_dataset",
    "ZebraConfig",
    "ZebraCurriculum",
    "ZebraDataset",
    "SelfReferenceCurriculum",
    "SelfReferenceConfig",
    "SelfReferenceDataset",
    "CircuitLogicConfig",
    "CircuitLogicDataset",
    "KnightsKnavesConfig",
    "KnightsKnavesDataset",
]
