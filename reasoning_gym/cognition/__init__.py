"""
Cognition tasks for training reasoning capabilities:
- Pattern recognition
- Sequence completion
- Logical reasoning
- Working memory
"""

from .number_sequences import NumberSequenceConfig, NumberSequenceDataset, sequence_dataset

__all__ = ["NumberSequenceDataset", "NumberSequenceConfig", "sequence_dataset"]
