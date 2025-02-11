"""Perform average / max pooling on a matrix"""

from copy import deepcopy
from dataclasses import dataclass
from random import Random
from typing import Dict, Optional

import numpy as np

from ..factory import ProceduralDataset, register_dataset

QUESTION_TEMPLATE = """Perform {pool_type} pooling on the following matrix:
{matrix}
"""


@dataclass
class PoolMatrixConfig:
    """Configuration for Pool Matrix dataset generation"""

    max_rows: int = 10  # Maximum rows of the matrix
    max_cols: int = 10  # Maximum columns of the matrix
    max_pool_size: int = 3  # Maximum pooling size

    size: int = 500  # Virtual dataset size
    seed: Optional[int] = None

    def validate(self):
        """Validate configuration parameters"""
        assert 1 <= self.max_rows, "max_rows must be at least 1"
        assert 1 <= self.max_cols, "max_cols must be at least 1"
        assert 1 <= self.max_pool_size, "max_pool_size must be at least 1"


class PoolMatrixDataset(ProceduralDataset):
    """Generates Pool Matrix exercises with configurable difficulty"""

    def __init__(self, config: PoolMatrixConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _get_matrix(self, rng: Random) -> np.ndarray:
        """Generate a random matrix"""
        rows = rng.randint(1, self.config.max_rows)
        cols = rng.randint(1, self.config.max_cols)
        return np.array([[rng.randint(0, 10) for _ in range(cols)] for _ in range(rows)])

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

    def score_answer(self, answer: Optional[str], entry: Dict[str, any]) -> float:
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

        matrix = self._get_matrix(rng)
        matrix_str = self._matrix_to_str(matrix)

        pool_size = rng.randint(1, self.config.max_pool_size)
        pool_type = rng.choice(["average", "max"])

        answer = self._average_pool(matrix, pool_size) if pool_type == "average" else self._max_pool(matrix, pool_size)
        answer_str = self._matrix_to_str(answer)

        return {
            "question": QUESTION_TEMPLATE.format(matrix=matrix_str, pool_type=pool_type),
            "answer": answer_str,
            "metadata": {
                "matrix": matrix.tolist(),
                "pool_type": pool_type,
                "pool_size": pool_size,
                "solution": answer.tolist(),
            },
        }


register_dataset("pool_matrix", PoolMatrixDataset, PoolMatrixConfig)
