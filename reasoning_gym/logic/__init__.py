"""
Logic tasks for training reasoning capabilities.
"""

from .aiw import AliceInWonderlandConfig, AliceInWonderlandCurriculum, AliceInWonderlandDataset
from .circuit_logic import CircuitLogicConfig, CircuitLogicCurriculum, CircuitLogicDataset
from .knights_knaves import KnightsKnavesConfig, KnightsKnavesCurriculum, KnightsKnavesDataset
from .propositional_logic import PropositionalLogicConfig, PropositionalLogicCurriculum, PropositionalLogicDataset
from .self_reference import SelfReferenceConfig, SelfReferenceCurriculum, SelfReferenceDataset
from .syllogisms import SyllogismConfig, SyllogismCurriculum, SyllogismDataset
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
    "SyllogismCurriculum",
    "syllogism_dataset",
    "ZebraConfig",
    "ZebraCurriculum",
    "ZebraDataset",
    "SelfReferenceCurriculum",
    "SelfReferenceConfig",
    "SelfReferenceDataset",
    "CircuitLogicConfig",
    "CircuitLogicDataset",
    "CircuitLogicCurriculum",
    "KnightsKnavesConfig",
    "KnightsKnavesDataset",
    "KnightsKnavesCurriculum",
]
