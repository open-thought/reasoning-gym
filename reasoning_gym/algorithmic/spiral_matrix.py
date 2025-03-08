"""Print elements of a matrix in spiral order.

A popular Leetcode problem:
https://leetcode.com/problems/spiral-matrix/description/
"""

from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from ..coaching import AttributeType, BaseCurriculum, RangeAttributeDefinition
from ..factory import ProceduralDataset, register_dataset

QUESTION_TEMPLATE = """Given a matrix, your job is to generate a list of elements in spiral order, starting from the top-left element.

The spiral order is clockwise, starting from the top-left corner. More precisely:
- Start from the top-left corner and move right.
- Move down towards the bottom-right corner.
- Move left towards the bottom-left corner.
- Move up towards the top-right corner.
- Repeat the steps for the inner elements of the matrix until every entry is visited.

Your output should be a space-separated list of integers, e.g. 1 2 3 4 5 6

For the matrix below, what is the list of elements in spiral order?
{matrix}
"""


@dataclass
class SpiralMatrixConfig:
    """Configuration for Spiral Matrix dataset generation"""

    min_n: int = 2  # Minimum number of rows/cols in the matrix
    max_n: int = 10  # Maximum number of rows/cols in the matrix

    size: int = 500  # Virtual dataset size
    seed: Optional[int] = None

    def validate(self):
        """Validate configuration parameters"""
        assert 2 <= self.min_n <= self.max_n, "min_n must be between 2 and max_n"


class SpiralMatrixDataset(ProceduralDataset):
    """Generates Spiral Matrix exercises with configurable difficulty"""

    def __init__(self, config: SpiralMatrixConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _get_matrix(self, rng: Random, n: int) -> list[list[int]]:
        """Generate a random matrix"""
        numbers = [rng.randint(0, 9) for _ in range(n**2)]
        rng.shuffle(numbers)
        matrix = [numbers[i * n : (i + 1) * n] for i in range(n)]
        return matrix

    def _get_spiral(self, matrix: list[list[int]]) -> list[int]:
        """Return the elements of the matrix in spiral order"""
        t, b = 0, len(matrix)
        l, r = 0, len(matrix[0])

        out = []

        while True:
            for i in range(l, r):
                out.append(matrix[t][i])
            t += 1
            if t == b:
                break

            for i in range(t, b):
                out.append(matrix[i][r - 1])
            r -= 1
            if l == r:
                break

            for i in range(r - 1, l - 1, -1):
                out.append(matrix[b - 1][i])
            b -= 1
            if t == b:
                break

            for i in range(b - 1, t - 1, -1):
                out.append(matrix[i][l])
            l += 1
            if l == r:
                break

        return out

    def _matrix_to_str(self, matrix: list[list[int]]) -> str:
        """Get a string representation of the matrix"""
        return "\n".join(" ".join(str(x) for x in row) for row in matrix)

    def _list_to_str(self, array: list[int]) -> str:
        """Get a string representation of the array"""
        return " ".join(str(x) for x in array)

    def __getitem__(self, idx: int) -> dict:
        """Generate a single Spiral Matrix question"""
        rng = Random(self.seed + idx)

        n = rng.randint(2, self.config.max_n)
        matrix = self._get_matrix(rng, n)
        matrix_str = self._matrix_to_str(matrix)
        answer = self._get_spiral(matrix)
        answer_str = self._list_to_str(answer)

        return {
            "question": QUESTION_TEMPLATE.format(matrix=matrix_str),
            "answer": answer_str,
            "metadata": {
                "matrix": matrix,
                "solution": answer,
                "difficulty": {"n": n},
            },
        }

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """Overwrite this method in derived classes if a single oracle answer is not available."""
        oracle_answer = entry["answer"].strip()

        if answer is not None and len(answer) > 0:
            answer = answer.strip()

            # Exact match
            if answer == oracle_answer:
                return 1.0

            # Try to see if the model's answer is a python list
            try:
                answer = " ".join(str(item) for item in eval(answer))
                if answer == oracle_answer:
                    return 0.1
            except Exception:
                pass

        return 0.0


class SpiralMatrixCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(SpiralMatrixCurriculum.__name__, SpiralMatrixConfig)

        # Define attributes
        self._define_attributes(
            RangeAttributeDefinition(
                name="n",
                levels=[10, 25, 50, 100],
                default_level=0,
                description="Number of rows/cols in the matrix",
                attr_type=AttributeType.APPEND,
                min_value=2,
                lower_field_name="min_n",
                upper_field_name="max_n",
            )
        )


register_dataset("spiral_matrix", SpiralMatrixDataset, SpiralMatrixConfig, SpiralMatrixCurriculum)
