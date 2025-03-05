"""Manipulate matrices by performing augmentations such as rotations, flips, mapping, etc."""

from copy import deepcopy
from dataclasses import dataclass
from random import Random
from typing import Any, Optional

import numpy as np

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

    min_rows: int = 2  # Minimum number of rows
    min_cols: int = 2  # Minimum number of columns
    max_rows: int = 10  # Maximum number of rows
    max_cols: int = 10  # Maximum number of columns
    min_transforms: int = 1  # Minimum number of transformations to apply
    max_transforms: int = 10  # Maximum number of transformations to apply
    w_rotate: float = 1  # Weight of rotating the matrix
    w_hmirror: float = 1  # Weight of horizontally mirroring the matrix
    w_vmirror: float = 1  # Weight of vertically mirroring the matrix
    w_dmirror: float = 1  # Weight of mirroring along the diagonal
    w_cmirror: float = 1  # Weight of mirroring along the counterdiagonal
    w_map: float = 1  # Weight of mapping a certain value to another
    w_crop: float = 1  # Weight of cropping the matrix
    w_remove_every_nth_row: float = 1  # Weight of removing every nth row
    w_remove_every_nth_col: float = 1  # Weight of removing every nth column
    w_zero_divisible: float = 1  # Weight of setting elements divisible by some number to zero

    size: int = 500  # Virtual dataset size
    seed: Optional[int] = None

    def validate(self):
        """Validate configuration parameters"""
        assert 1 <= self.min_rows, "min_rows must be at least 1"
        assert 1 <= self.min_cols, "min_cols must be at least 1"
        assert self.min_rows <= self.max_rows, "max_rows must be at least min_rows"
        assert self.min_cols <= self.max_cols, "max_cols must be at least min_cols"
        assert 1 <= self.min_transforms, "min_transforms must be at least 1"
        assert self.min_transforms <= self.max_transforms, "max_transforms must be at least min_transforms"
        assert (
            np.sum(
                np.exp(
                    [
                        self.w_rotate,
                        self.w_hmirror,
                        self.w_vmirror,
                        self.w_dmirror,
                        self.w_cmirror,
                        self.w_map,
                        self.w_crop,
                        self.w_remove_every_nth_row,
                        self.w_remove_every_nth_col,
                        self.w_zero_divisible,
                    ]
                )
            )
            > 0
        ), "At least one weight must be non-zero"


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
        weights = np.array(
            [
                config.w_rotate,
                config.w_hmirror,
                config.w_vmirror,
                config.w_dmirror,
                config.w_cmirror,
                config.w_map,
                config.w_crop,
                config.w_remove_every_nth_row,
                config.w_remove_every_nth_col,
                config.w_zero_divisible,
            ]
        )
        self._weights = np.exp(weights) / np.sum(np.exp(weights))

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

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        oracle_answer = entry["answer"].strip()
        if answer is not None and len(answer) > 0:
            answer = answer.strip()
            if answer == oracle_answer:
                return 1.0

            # perhaps the model's answer has unnecessary spaces (e.g. after last row element)
            answer = self._matrix_to_str([row.strip().split() for row in answer.strip().split("\n")]).strip()
            if answer == oracle_answer:
                return 1.0

            if oracle_answer in answer:
                return len(oracle_answer) / len(answer)

        return 0.0

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

        num_transforms = rng.randint(self.config.min_transforms, self.config.max_transforms)
        operations = []
        answer = deepcopy(matrix)

        while len(operations) < num_transforms:
            # Choose a transform randomly, weighted by the probability of each transform
            transform = rng.choices(self._all_transforms, weights=self._weights, k=1)[0]

            # Rotate
            if transform == "rotate":
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
            if transform == "hmirror":
                answer = self._hmirror(answer)
                operations.append({"transform": transform, "instruction": "- Horizontally mirror the matrix"})
            # Vertical mirror
            if transform == "vmirror":
                answer = self._vmirror(answer)
                operations.append({"transform": transform, "instruction": "- Vertically mirror the matrix"})
            # Diagonal mirror
            if transform == "dmirror":
                answer = self._dmirror(answer)
                operations.append({"transform": transform, "instruction": "- Mirror the matrix along the diagonal"})
            # Counterdiagonal mirror
            if transform == "cmirror":
                answer = self._cmirror(answer)
                operations.append(
                    {"transform": transform, "instruction": "- Mirror the matrix along the counterdiagonal"}
                )
            # Map a value to another
            if transform == "map":
                a, b = rng.sample(range(10), 2)
                answer = self._map(answer, a, b)
                operations.append(
                    {"transform": transform, "from": a, "to": b, "instruction": f"- Map each occurrence of {a} to {b}"}
                )
            # Set elements divisible by k to zero
            if transform == "zero_divisible":
                k = rng.randint(1, 9)
                answer = self._zero_divisible(answer, k)
                operations.append(
                    {"transform": transform, "k": k, "instruction": f"- Set all elements divisible by {k} to zero"}
                )
            # Crop the matrix
            if transform == "crop":
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
            if transform == "remove_every_nth_row" and num_rows(answer) > 1:
                n = rng.randint(2, num_rows(answer))
                answer = self._remove_every_nth_row(answer, n)
                formatting = "st" if n == 1 else "nd" if n == 2 else "th"
                operations.append(
                    {"transform": transform, "n": n, "instruction": f"- Remove every {n}-{formatting} row (1-indexed)"}
                )
            # Remove every nth column
            if transform == "remove_every_nth_col" and num_cols(answer) > 1:
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
