"""
Cognition tasks for training reasoning capabilities.
"""

from .arc_1d import Arc1DConfig, Arc1DDataset
from .color_cube_rotation import ColorCubeRotationConfig, ColorCubeRotationDataset
from .figlet_fonts import FigletFontConfig, FigletFontDataset
from .number_sequences import NumberSequenceConfig, NumberSequenceDataset
from .rubiks_cube import RubiksCubeConfig, RubiksCubeDataset

__all__ = [
    "Arc1DConfig",
    "Arc1DDataset",
    "ColorCubeRotationConfig",
    "ColorCubeRotationDataset",
    "FigletFontConfig",
    "FigletFontDataset",
    "NumberSequenceConfig",
    "NumberSequenceDataset",
    "RubiksCubeConfig",
    "RubiksCubeDataset",
]
