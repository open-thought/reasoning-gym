"""
Cognition tasks for training reasoning capabilities:
- Pattern recognition
- Sequence completion
- Logical reasoning
- Working memory
"""

from .color_cube_rotation import ColorCubeRotationConfig, ColorCubeRotationDataset, color_cube_rotation_dataset
from .number_sequences import NumberSequenceConfig, NumberSequenceDataset, number_sequence_dataset

__all__ = [
    "NumberSequenceConfig",
    "NumberSequenceDataset",
    "number_sequence_dataset",
    "ColorCubeRotationConfig",
    "ColorCubeRotationDataset",
    "color_cube_rotation_dataset",
]
