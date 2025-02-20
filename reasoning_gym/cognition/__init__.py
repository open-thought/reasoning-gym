"""
Cognition tasks for training reasoning capabilities.
"""

from .color_cube_rotation import ColorCubeRotationConfig, ColorCubeRotationDataset
from .figlet_fonts import FigletFontConfig, FigletFontDataset
from .needle_haystack import NeedleHaystackConfig, NeedleHaystackDataset
from .number_sequences import NumberSequenceConfig, NumberSequenceDataset
from .rectangle_count import RectangleCountConfig, RectangleCountDataset
from .rubiks_cube import RubiksCubeConfig, RubiksCubeDataset

__all__ = [
    "ColorCubeRotationConfig",
    "ColorCubeRotationDataset",
    "FigletFontConfig",
    "FigletFontDataset",
    "NumberSequenceConfig",
    "NumberSequenceDataset",
    "RubiksCubeConfig",
    "RubiksCubeDataset",
    "RectangleCountConfig",
    "RectangleCountDataset",
    "NeedleHaystackConfig",
    "NeedleHaystackDataset",
]
