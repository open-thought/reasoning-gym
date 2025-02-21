"""Perform average / max pooling on a matrix"""

from dataclasses import dataclass
from random import Random
from typing import Any, Optional

import numpy as np

from ..factory import ProceduralDataset, register_dataset

QUESTION_TEMPLATE = """Your job is to perform max/average pooling on the given matrix.
The stride is equal to the kernel size, meaning there is no overlap between the pooling regions.

Example 1:
- Input: Perform max pooling on the following matrix with a kernel size of 2:
1 2 3 4
5 6 7 8
9 10 11 12
13 14 15 16
- Output:
6 8
14 16

Example 2:
- Input: Perform average pooling on the following matrix with a kernel size of 2:
1 2 3 4
5 6 7 8
9 10 11 12
13 14 15 16
- Output:
3.5 5.5
11.5 13.5

Perform {pool_type} pooling on the following matrix with a kernel size of {pool_size}:
{matrix}
"""


@dataclass
class PoolMatrixConfig:
    """Configuration for Pool Matrix dataset generation"""

    min_rows: int = 2  # Minimum rows of the matrix
    min_cols: int = 2  # Minimum columns of the matrix
    max_rows: int = 10  # Maximum rows of the matrix
    max_cols: int = 10  # Maximum columns of the matrix
    max_pool_size: int = 3  # Maximum pooling size

    size: int = 500  # Virtual dataset size
    seed: Optional[int] = None

    def validate(self):
        """Validate configuration parameters"""
        assert 2 <= self.min_rows, "min_rows must be at least 2"
        assert 2 <= self.min_cols, "min_cols must be at least 2"
        assert self.min_rows <= self.max_rows, "max_rows must be at least min_rows"
        assert self.min_cols <= self.max_cols, "max_cols must be at least min_cols"
        assert 1 <= self.max_pool_size, "max_pool_size must be at least 1"


class PoolMatrixDataset(ProceduralDataset):
    """Generates Pool Matrix exercises with configurable difficulty"""

    def __init__(self, config: PoolMatrixConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _get_matrix(self, rng: Random) -> np.ndarray:
        """Generate a random matrix"""
        rows = rng.randint(self.config.min_rows, self.config.max_rows)
        cols = rng.randint(self.config.min_rows, self.config.max_cols)
        return np.random.randint(0, 10, (rows, cols))

    def _matrix_to_str(self, matrix: np.ndarray) -> str:
        """Get a string representation of the matrix"""
        return "\n".join(" ".join(str(round(x, 2)) for x in row) for row in matrix)

    def _max_pool(self, matrix: np.ndarray, pool_size: int) -> np.ndarray:
        """Perform max pooling on the matrix"""
        rows, cols = matrix.shape
        return np.array(
            [
                [np.max(matrix[i : i + pool_size, j : j + pool_size]) for j in range(0, cols, pool_size)]
                for i in range(0, rows, pool_size)
            ]
        )

    def _average_pool(self, matrix: np.ndarray, pool_size: int) -> np.ndarray:
        """Perform average pooling on the matrix"""
        rows, cols = matrix.shape
        return np.array(
            [
                [np.mean(matrix[i : i + pool_size, j : j + pool_size]) for j in range(0, cols, pool_size)]
                for i in range(0, rows, pool_size)
            ]
        )

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """Score the answer based on the metadata"""

        reward = 0.0
        try:
            if answer is not None:
                oracle_answer = np.array(entry["answer"])
                answer = np.array(answer)
                if oracle_answer.shape == answer.shape and np.allclose(oracle_answer, answer):
                    reward = 1.0
                if oracle_answer.shape == answer.shape:
                    reward = 0.1
                else:
                    reward = 0.01
        except:
            print("Error in scoring answer for Pool Matrix")
        return reward

    def __getitem__(self, idx: int) -> dict:
        """Generate a single Pool Matrix question"""
        rng = Random(self.seed + idx)
        np.random.seed(self.seed + idx)

        matrix = self._get_matrix(rng)
        matrix_str = self._matrix_to_str(matrix)

        pool_size = rng.randint(1, self.config.max_pool_size)
        pool_type = rng.choice(["average", "max"])

        answer = self._average_pool(matrix, pool_size) if pool_type == "average" else self._max_pool(matrix, pool_size)
        answer_str = self._matrix_to_str(answer)

        return {
            "question": QUESTION_TEMPLATE.format(matrix=matrix_str, pool_type=pool_type, pool_size=pool_size),
            "answer": answer_str,
            "metadata": {
                "matrix": matrix.tolist(),
                "pool_type": pool_type,
                "pool_size": pool_size,
                "solution": answer.tolist(),
            },
        }


register_dataset("pool_matrix", PoolMatrixDataset, PoolMatrixConfig)
