"""Manipulate matrices by performing augmentations such as rotations, flips, mapping, etc."""

from copy import deepcopy
from dataclasses import dataclass
from random import Random
from typing import Optional

from ..factory import ProceduralDataset, register_dataset

QUESTION_TEMPLATE = """For the following matrix:
{matrix}

Perform the following series of operations in order:
- Identity transformation, i.e. no change
{operations}
"""


def num_rows(matrix: list[list[int]]) -> int:
    return len(matrix)


def num_cols(matrix: list[list[int]]) -> int:
    return len(matrix[0]) if matrix else 0


@dataclass
class ManipulateMatrixConfig:
    """Configuration for Manipulate Matrix dataset generation"""

    min_rows: int = 1  # Minimum number of rows
    min_cols: int = 1  # Minimum number of columns
    max_rows: int = 10  # Maximum number of rows
    max_cols: int = 10  # Maximum number of columns
    max_transforms: int = 5  # Maximum number of transformations to apply
    p_rotate: float = 0.2  # Probability of rotating the matrix
    p_hmirror: float = 0.2  # Probability of horizontally mirroring the matrix
    p_vmirror: float = 0.2  # Probability of vertically mirroring the matrix
    p_dmirror: float = 0.2  # Probability of mirroring along the diagonal
    p_cmirror: float = 0.2  # Probability of mirroring along the counterdiagonal
    p_map: float = 0.2  # Probability of mapping a certain value to another
    p_crop: float = 0.2  # Probability of cropping the matrix
    p_remove_every_nth_row: float = 0.2  # Probability of removing every nth row
    p_remove_every_nth_col: float = 0.2  # Probability of removing every nth column
    p_zero_divisible: float = 0.2  # Probability of setting elements divisible by some number to zero

    size: int = 500  # Virtual dataset size
    seed: Optional[int] = None

    def validate(self):
        """Validate configuration parameters"""
        assert 1 <= self.min_rows, "min_rows must be at least 1"
        assert 1 <= self.min_cols, "min_cols must be at least 1"
        assert self.min_rows <= self.max_rows, "max_rows must be at least min_rows"
        assert self.min_cols <= self.max_cols, "max_cols must be at least min_cols"
        assert 0 <= self.max_transforms, "max_transforms must be non-negative"
        assert 0 <= self.p_rotate <= 1, "p_rotate must be between 0 and 1"
        assert 0 <= self.p_hmirror <= 1, "p_hmirror must be between 0 and 1"
        assert 0 <= self.p_vmirror <= 1, "p_vmirror must be between 0 and 1"
        assert 0 <= self.p_dmirror <= 1, "p_dmirror must be between 0 and 1"
        assert 0 <= self.p_cmirror <= 1, "p_cmirror must be between 0 and 1"
        assert 0 <= self.p_map <= 1, "p_map must be between 0 and 1"
        assert 0 <= self.p_crop <= 1, "p_crop must be between 0 and 1"
        assert 0 <= self.p_remove_every_nth_row <= 1, "p_remove_every_nth_row must be between 0 and 1"
        assert 0 <= self.p_remove_every_nth_col <= 1, "p_remove_nth_col must be between 0 and 1"
        assert 0 <= self.p_zero_divisible <= 1, "p_zero_divisible must be between 0 and 1"


class ManipulateMatrixDataset(ProceduralDataset):
    """Generates Manipulate Matrix exercises with configurable difficulty"""

    def __init__(self, config: ManipulateMatrixConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)
        self._rotations = {
            "90": self._rot90,
            "180": self._rot180,
            "270": self._rot270,
            "360": self._identity,
        }
        self._all_transforms = [
            "rotate",
            "hmirror",
            "vmirror",
            "dmirror",
            "cmirror",
            "map",
            "zero_divisible",
            "crop",
            "remove_every_nth_row",
            "remove_every_nth_col",
        ]

    def _get_matrix(self, rng: Random) -> list[list[int]]:
        """Generate a random matrix"""
        rows = rng.randint(self.config.min_rows, self.config.max_rows)
        cols = rng.randint(self.config.min_cols, self.config.max_cols)
        numbers = [rng.randint(0, 9) for _ in range(rows * cols)]
        matrix = [numbers[i * cols : (i + 1) * cols] for i in range(rows)]
        return matrix

    def _matrix_to_str(self, matrix: list[list[int]]) -> str:
        """Get a string representation of the matrix"""
        return "\n".join(" ".join(str(x) for x in row) for row in matrix)

    def _identity(self, matrix: list[list[int]]) -> list[list[int]]:
        """Identity transformation"""
        return matrix

    def _rot90(self, matrix: list[list[int]]) -> list[list[int]]:
        """quarter clockwise rotation"""
        return [list(row) for row in zip(*matrix[::-1])]

    def _rot180(self, matrix: list[list[int]]) -> list[list[int]]:
        """half rotation"""
        return [list(row[::-1]) for row in matrix[::-1]]

    def _rot270(self, matrix: list[list[int]]) -> list[list[int]]:
        """quarter anticlockwise rotation"""
        return [list(row[::-1]) for row in zip(*matrix[::-1])][::-1]

    def _hmirror(self, matrix: list[list[int]]) -> list[list[int]]:
        """mirroring along horizontal"""
        return matrix[::-1]

    def _vmirror(self, matrix: list[list[int]]) -> list[list[int]]:
        """mirroring along vertical"""
        return [row[::-1] for row in matrix]

    def _dmirror(self, matrix: list[list[int]]) -> list[list[int]]:
        """mirroring along diagonal"""
        return list(list(row) for row in zip(*matrix))

    def _cmirror(self, matrix: list[list[int]]) -> list[list[int]]:
        """mirroring along counterdiagonal"""
        return list(list(row) for row in zip(*[r[::-1] for r in matrix[::-1]]))

    def _map(self, matrix: list[list[int]], a: int, b: int) -> list[list[int]]:
        """mapping a to b"""
        return [[b if x == a else x for x in row] for row in matrix]

    def _zero_divisible(self, matrix: list[list[int]], k: int) -> list[list[int]]:
        """set elements divisible by k to zero"""
        return [[0 if x % k == 0 else x for x in row] for row in matrix]

    def _crop(
        self, matrix: list[list[int]], row_start: int, row_end: int, col_start: int, col_end: int
    ) -> list[list[int]]:
        """crop the matrix (1-indexed)"""
        return [row[col_start - 1 : col_end] for row in matrix[row_start - 1 : row_end]]

    def _remove_every_nth_row(self, matrix: list[list[int]], n: int) -> list[list[int]]:
        """remove every nth row (1-indexed)"""
        return [row for i, row in enumerate(matrix, start=1) if i % n != 0]

    def _remove_every_nth_col(self, matrix: list[list[int]], n: int) -> list[list[int]]:
        """remove every nth column (1-indexed)"""
        return [[col for i, col in enumerate(row, start=1) if i % n != 0] for row in matrix]

    def __getitem__(self, idx: int) -> dict:
        """Generate a single Manipulate Matrix question"""
        rng = Random(self.seed + idx)

        matrix = self._get_matrix(rng)
        matrix_str = self._matrix_to_str(matrix)

        num_transforms = rng.randint(0, self.config.max_transforms)
        transforms = rng.sample(self._all_transforms, num_transforms)
        operations = []

        answer = deepcopy(matrix)

        for transform in transforms:
            # Rotate
            if transform == "rotate" and rng.random() < self.config.p_rotate:
                rotation = rng.choice(list(self._rotations.keys()))
                answer = self._rotations[rotation](answer)
                operations.append(
                    {
                        "transform": transform,
                        "degrees": rotation,
                        "instruction": f"- Rotate the matrix {rotation} degrees",
                    }
                )
            # Horizontal mirror
            if transform == "hmirror" and rng.random() < self.config.p_hmirror:
                answer = self._hmirror(answer)
                operations.append({"transform": transform, "instruction": "- Horizontally mirror the matrix"})
            # Vertical mirror
            if transform == "vmirror" and rng.random() < self.config.p_vmirror:
                answer = self._vmirror(answer)
                operations.append({"transform": transform, "instruction": "- Vertically mirror the matrix"})
            # Diagonal mirror
            if transform == "dmirror" and rng.random() < self.config.p_dmirror:
                answer = self._dmirror(answer)
                operations.append({"transform": transform, "instruction": "- Mirror the matrix along the diagonal"})
            # Counterdiagonal mirror
            if transform == "cmirror" and rng.random() < self.config.p_cmirror:
                answer = self._cmirror(answer)
                operations.append(
                    {"transform": transform, "instruction": "- Mirror the matrix along the counterdiagonal"}
                )
            # Map a value to another
            if transform == "map" and rng.random() < self.config.p_map:
                a, b = rng.sample(range(10), 2)
                answer = self._map(answer, a, b)
                operations.append(
                    {"transform": transform, "from": a, "to": b, "instruction": f"- Map each occurrence of {a} to {b}"}
                )
            # Set elements divisible by k to zero
            if transform == "zero_divisible" and rng.random() < self.config.p_zero_divisible:
                k = rng.randint(1, 9)
                answer = self._zero_divisible(answer, k)
                operations.append(
                    {"transform": transform, "k": k, "instruction": f"- Set all elements divisible by {k} to zero"}
                )
            # Crop the matrix
            if transform == "crop" and rng.random() < self.config.p_crop:
                row_start = rng.randint(1, num_rows(answer))
                row_end = rng.randint(row_start, num_rows(answer))
                col_start = rng.randint(1, num_cols(answer))
                col_end = rng.randint(col_start, num_cols(answer))
                answer = self._crop(answer, row_start, row_end, col_start, col_end)
                operations.append(
                    {
                        "transform": transform,
                        "row_start": row_start,
                        "row_end": row_end,
                        "col_start": col_start,
                        "col_end": col_end,
                        "instruction": f"- Crop the matrix to rows {row_start}-{row_end} and columns {col_start}-{col_end} (1-indexed)",
                    }
                )
            # Remove every nth row
            if (
                transform == "remove_every_nth_row"
                and rng.random() < self.config.p_remove_every_nth_row
                and num_rows(answer) > 1
            ):
                n = rng.randint(2, num_rows(answer))
                answer = self._remove_every_nth_row(answer, n)
                formatting = "st" if n == 1 else "nd" if n == 2 else "th"
                operations.append(
                    {"transform": transform, "n": n, "instruction": f"- Remove every {n}-{formatting} row (1-indexed)"}
                )
            # Remove every nth column
            if (
                transform == "remove_every_nth_col"
                and rng.random() < self.config.p_remove_every_nth_col
                and num_cols(answer) > 1
            ):
                n = rng.randint(2, num_cols(answer))
                answer = self._remove_every_nth_col(answer, n)
                formatting = "st" if n == 1 else "nd" if n == 2 else "th"
                operations.append(
                    {
                        "transform": transform,
                        "n": n,
                        "instruction": f"- Remove every {n}-{formatting} column (1-indexed)",
                    }
                )

        answer_str = self._matrix_to_str(answer)

        return {
            "question": QUESTION_TEMPLATE.format(
                matrix=matrix_str, operations="\n".join(op["instruction"] for op in operations)
            ),
            "answer": answer_str,
            "metadata": {"matrix": matrix, "solution": answer, "operations": operations},
        }


register_dataset("manipulate_matrix", ManipulateMatrixDataset, ManipulateMatrixConfig)
