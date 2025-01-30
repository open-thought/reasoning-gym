"""
Cognition tasks for training reasoning capabilities:
- Pattern recognition
- Sequence completion
- Logical reasoning
- Working memory
"""

from .color_cube_rotation import ColorCubeRotationConfig, ColorCubeRotationDataset
from .figlet_fonts import FigletFontConfig, FigletFontDataset
from .number_sequences import NumberSequenceConfig, NumberSequenceDataset
from .rubiks_cube import RubiksCubeConfig, RubiksCubeDataset

__all__ = [
    "NumberSequenceConfig",
    "NumberSequenceDataset",
    "ColorCubeRotationConfig",
    "ColorCubeRotationDataset",
    "RubiksCubeConfig",
    "RubiksCubeDataset",
    "FigletFontConfig",
    "FigletFontDataset",
]
