import json
from dataclasses import dataclass
from random import Random
from typing import Any, Optional

import cellpylib as cpl

from ..coaching import BaseCurriculum, ScalarAttributeDefinition
from ..factory import ProceduralDataset, register_dataset

DATASET_NAME = "game_of_life"


@dataclass
class GameOfLifeConfig:
    """Configuration for Game of Life puzzle generation"""

    grid_size_x: int = 10
    grid_size_y: int = 10
    filled_cells_weights: float = 0.1
    filled_cells: Optional[int] = None  # max live cells, defaults to filled_cells_weights * x * y
    simulation_steps: int = 1
    seed: Optional[int] = None
    size: int = 500

    def __post_init__(self):
        if self.filled_cells is None:
            self.filled_cells = int(self.filled_cells_weights * self.grid_size_x * self.grid_size_y)

    def validate(self):
        """Validate configuration parameters"""
        assert 3 <= self.grid_size_x <= 999, "grid_size_x must be between 0 and 999"
        assert 3 <= self.grid_size_y <= 999, "grid_size_y must be between 0 and 999"
        assert self.simulation_steps >= 0, "simulation_steps must be gte 0"
        assert self.filled_cells <= self.grid_size_x * self.grid_size_y, "filled_cells must fit in x times y"


class GameOfLifeDataset(ProceduralDataset):
    """Generates Game of Life games with configurable parameters"""

    def __init__(self, config: GameOfLifeConfig):
        self._prompt_templates = [
            "What will this Game of Life board look like after {simulation_steps} steps of simulation? Assume a Moore neighborhood and wrapping topology. Reply as array of arrays representing rows in the grid from top to bottom in JSON format. (An empty 3x3 grid would look like this: [[0,0,0],[0,0,0],[0,0,0]])\n\n{board}."
        ]

        super().__init__(config=config, seed=config.seed, size=config.size)

    def __getitem__(self, idx: int) -> dict:
        """Generate a single GameOfLife task

        Returns:
            dict with keys:
                - question: str, the task description
                - answer: str, a solution string
                - metadata: dict with generation parameters
        """
        rng = Random(self.seed + idx)

        # Make the board
        board = cpl.init_simple2d(self.config.grid_size_x, self.config.grid_size_y)
        board[:, :, :] = 0

        # Add the cells
        for i in range(0, self.config.filled_cells):
            rx = rng.randint(0, self.config.grid_size_x - 1)
            ry = rng.randint(0, self.config.grid_size_y - 1)
            board[:, rx, ry] = 1

        # Simulate the result to get the answer
        evolved = cpl.evolve2d(
            board,
            timesteps=self.config.simulation_steps + 1,
            apply_rule=cpl.game_of_life_rule,
            memoize="recursive",
        )

        rows = [json.dumps(board[0, i].tolist(), separators=(",", ":")) for i in range(board.shape[1])]
        board_str = "[" + ",\n ".join(rows) + "]"

        final_step = evolved[-1]
        final_step_list = final_step.tolist()
        result_str = json.dumps(final_step_list, separators=(",", ":"))

        return {
            "question": rng.choice(self._prompt_templates).format(
                simulation_steps=self.config.simulation_steps, board=board_str
            ),
            "answer": result_str,
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "grid_size_x": self.config.grid_size_x,
                "grid_size_y": self.config.grid_size_y,
                "filled_cells": self.config.filled_cells,
                "simulation_steps": self.config.simulation_steps,
                "board": board[0].tolist(),
                "difficulty": {
                    "grid_size_x": self.config.grid_size_x,
                    "grid_size_y": self.config.grid_size_y,
                    "filled_cells_weights": self.config.filled_cells_weights,
                    "simulation_steps": self.config.simulation_steps,
                },
            },
        }

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """Score a GoL answer.

        A correct grid scores 1.0. Otherwise the score is the share of cell errors the answer avoids
        compared to the best trivial guess (an all-dead grid or the unchanged input board), floored at 0.

        Args:
            answer (Optional[str]): The user's answer.
            entry (dict[str, Any]): The original dataset entry containing the correct answer.

        Returns:
            float: The computed score between 0.0 and 1.0.
        """

        if answer == None:
            return 0.0

        try:
            ans_arr = json.loads(answer)
            correct_arr = json.loads(entry["answer"])
        except Exception:
            return 0.0

        errors = _count_errors(ans_arr, correct_arr)
        if errors == 0:
            return 1.0

        # An all-dead guess is wrong on every live cell.
        baseline_errors = sum(value != 0 for row in correct_arr for value in row)
        if "board" in entry["metadata"]:
            baseline_errors = min(baseline_errors, _count_errors(entry["metadata"]["board"], correct_arr))
        if baseline_errors == 0:
            return 0.0

        return max(0.0, 1.0 - errors / baseline_errors)


def _count_errors(grid: Any, expected: list[list[int]]) -> int:
    """Count cells of `expected` that `grid` gets wrong; missing cells count as wrong."""
    errors = 0
    for i, row in enumerate(expected):
        for j, value in enumerate(row):
            try:
                errors += grid[i][j] != value
            except (IndexError, KeyError, TypeError):
                errors += 1
    return errors


class GameOfLifeCurriculum(BaseCurriculum):
    """Curriculum for Game of Life dataset"""

    def __init__(self):
        super().__init__(GameOfLifeCurriculum.__name__, GameOfLifeConfig)

        # Define attributes
        self._define_attributes(
            ScalarAttributeDefinition(
                name="grid_size_x",
                field_name="grid_size_x",
                levels=[10, 25, 50, 100],
                description="Grid size in the x direction",
            ),
            ScalarAttributeDefinition(
                name="grid_size_y",
                field_name="grid_size_y",
                levels=[10, 25, 50, 100],
                description="Grid size in the y direction",
            ),
            # Filled cells should be 10%, 20%, 30%, 50% of the grid_size_x * grid_size_y
            ScalarAttributeDefinition(
                name="filled_cells_weights",
                field_name="filled_cells_weights",
                levels=[0.1, 0.2, 0.5, 0.8],
                description="Percentage of filled cells in the grid",
            ),
            ScalarAttributeDefinition(
                name="simulation_steps",
                field_name="simulation_steps",
                levels=[1, 2, 5, 10],
                description="Number of simulation steps to run",
            ),
        )


register_dataset(DATASET_NAME, GameOfLifeDataset, GameOfLifeConfig, GameOfLifeCurriculum)
