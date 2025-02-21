"""Find the distance to the nearest 0 for each cell in a binary matrix.

A popular Leetcode problem:
https://leetcode.com/problems/01-matrix/description/
"""

from collections import deque
from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from ..factory import ProceduralDataset, register_dataset

QUESTION_TEMPLATE = """Given a square matrix, your job is to find the taxicab (Manhattan) distance of the nearest 0 for each cell.

Example:
- Input: Find the distance to the nearest 0 for each cell in the matrix below:
0 0 0
0 1 0
1 1 1
- Output:
0 0 0
0 1 0
1 2 1
- Explanation
    - Each cell with a 0 has a distance of 0 to itself.
    - The cell at (1, 1) has a distance of 1 to the nearest 0 (any of the three 0's at (1, 0), (0, 1), (1, 2)).
    - The cell at (2, 0) has a distance of 1 to the nearest 0 (the 0 at (1, 0)).
    - The cell at (2, 1) has a distance of 2 to the nearest 0 (any of the two 0's at (1, 0), (1, 2))
    - The cell at (2, 2) has a distance of 1 to the nearest 0 (the 0 at (1, 2)).
    - Hence, the final answer is the matrix is the output shown above, where each cell contains the distance to the nearest 0, in the same format as the input matrix.

Find the distance to the nearest 0 for each cell in the matrix below:
{matrix}
"""


@dataclass
class BinaryMatrixConfig:
    """Configuration for Binary Matrix dataset generation"""

    min_n: int = 3  # Minimum size of the matrix
    max_n: int = 10  # Maximum size of the matrix
    p_zero: float = 0.25  # Probability of a cell being 0

    size: int = 500  # Virtual dataset size
    seed: Optional[int] = None

    def validate(self):
        """Validate configuration parameters"""
        assert 1 <= self.min_n, "min_n must be at least 1"
        assert self.min_n <= self.max_n, "min_n must be less than or equal to max_n"
        assert 0 < self.p_zero <= 1, "p_zero must be between 0 and 1"


class BinaryMatrixDataset(ProceduralDataset):
    """Generates Binary Matrix exercises with configurable difficulty"""

    def __init__(self, config: BinaryMatrixConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _get_binary_matrix(self, rng: Random) -> list[list[int]]:
        """Generate a random binary matrix"""
        n = rng.randint(self.config.min_n, self.config.max_n)
        # Ensure at least one 0 in the matrix, so that a solution exists
        numbers = [0] + [0 if rng.random() < self.config.p_zero else 1 for _ in range(n**2 - 1)]
        rng.shuffle(numbers)
        matrix = [numbers[i * n : (i + 1) * n] for i in range(n)]
        return matrix

    def _get_distances(self, matrix: list[list[int]]) -> list[list[int]]:
        """Get the distance to the nearest 0 for each cell in the matrix"""
        n = len(matrix)
        directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        visited = set()
        queue = deque()

        output = [[float("inf")] * n for _ in range(n)]

        for r in range(n):
            for c in range(n):
                if matrix[r][c] == 0:
                    output[r][c] = 0
                    visited.add((r, c))
                    queue.append((r, c))

        clock = 1
        while True:
            temp = deque()
            while queue:
                r, c = queue.popleft()
                for dr, dc in directions:
                    new_r, new_c = r + dr, c + dc
                    if (
                        0 <= new_r < n
                        and 0 <= new_c < n
                        and (new_r, new_c) not in visited
                        and matrix[new_r][new_c] == 1
                    ):
                        output[new_r][new_c] = clock
                        visited.add((new_r, new_c))
                        temp.append((new_r, new_c))
            if temp:
                queue = temp
            else:
                break
            clock += 1

        return output

    def _matrix_to_str(self, matrix: list[list[int]]) -> str:
        """Get a string representation of the matrix"""
        return "\n".join(" ".join(str(x) for x in row) for row in matrix)

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """Overwrite this method in derived classes if a single oracle answer is not available."""
        oracle_answer = entry["answer"]
        if answer is not None:
            if answer == oracle_answer:
                return 1.0
            else:
                try:
                    # check if answer is python list of lists
                    answer = self._matrix_to_str(eval(answer))
                    if answer == oracle_answer:
                        return 0.5
                except Exception as e:
                    return 0.01
        return 0.0

    def __getitem__(self, idx: int) -> dict:
        """Generate a single Binary Matrix question"""
        rng = Random(self.seed + idx)

        matrix = self._get_binary_matrix(rng)
        matrix_str = self._matrix_to_str(matrix)

        answer = self._get_distances(matrix)
        answer_str = self._matrix_to_str(answer)

        return {
            "question": QUESTION_TEMPLATE.format(matrix=matrix_str),
            "answer": answer_str,
            "metadata": {"matrix": matrix, "solution": answer},
        }


register_dataset("binary_matrix", BinaryMatrixDataset, BinaryMatrixConfig)
