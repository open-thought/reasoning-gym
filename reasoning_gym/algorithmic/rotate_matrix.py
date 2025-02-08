"""Rotate a square matrix by 90 degrees clockwise.

A popular Leetcode problem:
https://leetcode.com/problems/rotate-image/description/
"""

from copy import deepcopy
from dataclasses import dataclass
from random import Random
from typing import Optional

from ..factory import ProceduralDataset, register_dataset

QUESTION_TEMPLATE = """Given a square matrix, your job is to rotate it by 90 degrees clockwise.

Example:

Input:
1 2 3
4 5 6
7 8 9

Output:
7 4 1
8 5 2
9 6 3

Rotate the matrix below by 90 degrees clockwise:
{matrix}
"""


@dataclass
class RotateMatrixConfig:
    """Configuration for Rotate Matrix dataset generation"""

    max_n: int = 10  # Maximum size of the matrix

    size: int = 500  # Virtual dataset size
    seed: Optional[int] = None

    def validate(self):
        """Validate configuration parameters"""
        assert 1 <= self.max_n, "max_n must be at least 1"


class RotateMatrixDataset(ProceduralDataset):
    """Generates Rotate Matrix exercises with configurable difficulty"""

    def __init__(self, config: RotateMatrixConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _get_matrix(self, rng: Random) -> list[list[int]]:
        """Generate a random matrix"""
        n = rng.randint(1, self.config.max_n)
        numbers = list(range(n**2))
        rng.shuffle(numbers)
        matrix = [numbers[i * n : (i + 1) * n] for i in range(n)]
        return matrix

    def _get_rotated(self, matrix: list[list[int]]) -> list[list[int]]:
        """Rotate the matrix by 90 degrees clockwise"""
        n = len(matrix)
        output = deepcopy(matrix)

        for l in range(n // 2):
            for i in range(l, n - 1 - l):
                (output[l][i], output[i][n - 1 - l], output[n - 1 - l][n - 1 - i], output[n - 1 - i][l]) = (
                    matrix[n - 1 - i][l],
                    matrix[l][i],
                    matrix[i][n - 1 - l],
                    matrix[n - 1 - l][n - 1 - i],
                )

        return output

    def _matrix_to_str(self, matrix: list[list[int]]) -> str:
        """Get a string representation of the matrix"""
        return "\n".join(" ".join(str(x) for x in row) for row in matrix)

    def __getitem__(self, idx: int) -> dict:
        """Generate a single Spiral Matrix question"""
        rng = Random(self.seed + idx)

        matrix = self._get_matrix(rng)
        matrix_str = self._matrix_to_str(matrix)
        answer = self._get_rotated(matrix)
        answer_str = self._matrix_to_str(answer)

        return {
            "question": QUESTION_TEMPLATE.format(matrix=matrix_str),
            "answer": answer_str,
            "metadata": {"matrix": matrix, "solution": answer},
        }


register_dataset("rotate_matrix", RotateMatrixDataset, RotateMatrixConfig)
