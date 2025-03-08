"""Rotate a square matrix clockwise.

A popular Leetcode problem:
https://leetcode.com/problems/rotate-image/description/
"""

from copy import deepcopy
from dataclasses import dataclass
from random import Random
from typing import Optional

from ..coaching import AttributeType, BaseCurriculum, RangeAttributeDefinition
from ..factory import ProceduralDataset, register_dataset

QUESTION_TEMPLATE = """Given a square matrix, your job is to rotate it clockwise.

Your output should be a matrix in the same format as the input.

Rotate the matrix below by {degrees} degrees clockwise:
{matrix}
"""


@dataclass
class RotateMatrixConfig:
    """Configuration for Rotate Matrix dataset generation"""

    min_n: int = 2  # Minimum size of the matrix
    max_n: int = 10  # Maximum size of the matrix
    min_rotations: int = 0  # Minimum number of rotations
    max_rotations: int = 10  # Maximum number of rotations (90 degrees each)

    size: int = 500  # Virtual dataset size
    seed: Optional[int] = None

    def validate(self):
        """Validate configuration parameters"""
        assert 2 <= self.min_n <= self.max_n, "min_n and max_n must be between 2 and 10"
        assert 0 <= self.min_rotations <= self.max_rotations, "min_rotations must be between 0 and max_rotations"


class RotateMatrixDataset(ProceduralDataset):
    """Generates Rotate Matrix exercises with configurable difficulty"""

    def __init__(self, config: RotateMatrixConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _get_matrix(self, rng: Random, n: int) -> list[list[int]]:
        """Generate a random matrix"""
        numbers = list(rng.randint(0, 9) for _ in range(n**2))
        matrix = [numbers[i * n : (i + 1) * n] for i in range(n)]
        return matrix

    def _rot90(self, matrix: list[list[int]]) -> list[list[int]]:
        """quarter clockwise rotation"""
        return [list(row) for row in zip(*matrix[::-1])]

    def _get_rotated(self, matrix: list[list[int]], num_rotations: int) -> list[list[int]]:
        """Rotate the matrix K times by 90 degrees clockwise"""
        num_rotations %= 4
        output = deepcopy(matrix)
        for _ in range(num_rotations):
            output = self._rot90(output)
        return output

    def _matrix_to_str(self, matrix: list[list[int]]) -> str:
        """Get a string representation of the matrix"""
        return "\n".join(" ".join(str(x) for x in row) for row in matrix)

    def __getitem__(self, idx: int) -> dict:
        """Generate a single Rotate Matrix question"""
        rng = Random(self.seed + idx)

        n = rng.randint(self.config.min_n, self.config.max_n)
        matrix = self._get_matrix(rng, n)
        num_rotations = rng.randint(self.config.min_rotations, self.config.max_rotations)
        matrix_str = self._matrix_to_str(matrix)

        answer = self._get_rotated(matrix, num_rotations)
        answer_str = self._matrix_to_str(answer)

        return {
            "question": QUESTION_TEMPLATE.format(matrix=matrix_str, degrees=num_rotations * 90),
            "answer": answer_str,
            "metadata": {
                "matrix": matrix,
                "num_rotations": num_rotations,
                "solution": answer,
                "difficulty": {
                    "n": n,
                    "num_rotations": num_rotations,
                },
            },
        }


class RotateMatrixCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(RotateMatrixCurriculum.__name__, RotateMatrixConfig)

        # Define attributes
        self._define_attributes(
            RangeAttributeDefinition(
                name="n",
                levels=[10, 25, 50, 100],
                default_level=0,
                description="Size of the square matrix",
                attr_type=AttributeType.APPEND,
                min_value=2,
                lower_field_name="min_n",
                upper_field_name="max_n",
            ),
            RangeAttributeDefinition(
                name="num_rotations",
                levels=[4, 8, 12, 16],
                default_level=0,
                description="Number of 90-degree rotations",
                attr_type=AttributeType.APPEND,
                min_value=0,
                lower_field_name="min_rotations",
                upper_field_name="max_rotations",
            ),
        )


register_dataset("rotate_matrix", RotateMatrixDataset, RotateMatrixConfig, RotateMatrixCurriculum)
