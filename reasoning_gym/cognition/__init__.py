"""
Cognition tasks for training reasoning capabilities.
"""

from .color_cube_rotation import ColorCubeRotationConfig, ColorCubeRotationDataset
from .figlet_fonts import FigletFontConfig, FigletFontDataset
from .modulo_grid import ModuloGridConfig, ModuloGridDataset
from .needle_haystack import NeedleHaystackConfig, NeedleHaystackCurriculum, NeedleHaystackDataset
from .number_sequences import NumberSequenceConfig, NumberSequenceCurriculum, NumberSequenceDataset
from .rectangle_count import RectangleCountConfig, RectangleCountCurriculum, RectangleCountDataset
from .rubiks_cube import RubiksCubeConfig, RubiksCubeDataset

__all__ = [
    "ColorCubeRotationConfig",
    "ColorCubeRotationDataset",
    "FigletFontConfig",
    "FigletFontDataset",
    "NumberSequenceConfig",
    "NumberSequenceDataset",
    "NumberSequenceCurriculum",
    "RubiksCubeConfig",
    "RubiksCubeDataset",
    "RectangleCountConfig",
    "RectangleCountCurriculum",
    "RectangleCountDataset",
    "NeedleHaystackConfig",
    "NeedleHaystackDataset",
    "NeedleHaystackCurriculum",
    "ModuloGridConfig",
    "ModuloGridDataset",
]
