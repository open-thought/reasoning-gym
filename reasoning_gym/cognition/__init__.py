"""
Cognition tasks for training reasoning capabilities:
- Pattern recognition
- Sequence completion
- Logical reasoning
- Working memory
"""

from .color_cube_rotation import ColorCubeRotationConfig, ColorCubeRotationDataset
from .number_sequences import NumberSequenceConfig, NumberSequenceDataset

__all__ = [
    "NumberSequenceConfig",
    "NumberSequenceDataset",
    "ColorCubeRotationConfig",
    "ColorCubeRotationDataset",
]
