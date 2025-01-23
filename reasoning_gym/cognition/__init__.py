"""
Cognition tasks for training reasoning capabilities:
- Pattern recognition
- Sequence completion
- Logical reasoning
- Working memory
"""

from .sequences import SequenceConfig, SequenceDataset, sequence_dataset

__all__ = ["SequenceDataset", "SequenceConfig", "sequence_dataset"]
