"""
Cognition tasks for training reasoning capabilities.
"""

from .color_cube_rotation import ColorCubeRotationConfig, ColorCubeRotationCurriculum, ColorCubeRotationDataset
from .figlet_fonts import FigletFontConfig, FigletFontCurriculum, FigletFontDataset
from .modulo_grid import ModuloGridConfig, ModuloGridDataset
from .needle_haystack import NeedleHaystackConfig, NeedleHaystackCurriculum, NeedleHaystackDataset
from .number_sequences import NumberSequenceConfig, NumberSequenceCurriculum, NumberSequenceDataset
from .pyraminx import PyraminxConfig, PyraminxCurriculum, PyraminxDataset
from .rectangle_count import RectangleCountConfig, RectangleCountCurriculum, RectangleCountDataset
from .rubiks_cube import RubiksCubeConfig, RubiksCubeCurriculum, RubiksCubeDataset

__all__ = [
    "ColorCubeRotationConfig",
    "ColorCubeRotationDataset",
    "ColorCubeRotationCurriculum",
    "FigletFontConfig",
    "FigletFontDataset",
    "FigletFontCurriculum",
    "NumberSequenceConfig",
    "NumberSequenceDataset",
    "NumberSequenceCurriculum",
    "PyraminxConfig",
    "PyraminxDataset",
    "PyraminxCurriculum",
    "RubiksCubeConfig",
    "RubiksCubeDataset",
    "RubiksCubeCurriculum",
    "RectangleCountConfig",
    "RectangleCountCurriculum",
    "RectangleCountDataset",
    "NeedleHaystackConfig",
    "NeedleHaystackDataset",
    "NeedleHaystackCurriculum",
    "ModuloGridConfig",
    "ModuloGridDataset",
]
