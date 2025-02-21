import json
from dataclasses import dataclass
from random import Random
from typing import Any, Optional

import cellpylib as cpl

from ..factory import ProceduralDataset, register_dataset


@dataclass
class GameOfLifeConfig:
    """Configuration for sudoku puzzle generation"""

    grid_size_x: int = 10
    grid_size_y: int = 10
    filled_cells: int = 100  # actually a max
    simulation_steps: int = 1
    seed: Optional[int] = None
    size: int = 500

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
            "What will this Game of Life board look like after {simulation_steps} steps of simulation? Reply as array of arrays representing rows in the grid from top to bottom in JSON format. (An empty 3x3 grid would look like this: [[0,0,0],[0,0,0],[0,0,0]])\n\n{board}."
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
                "grid_size_x": self.config.grid_size_x,
                "grid_size_y": self.config.grid_size_y,
                "filled_cells": self.config.filled_cells,
                "simulation_steps": self.config.simulation_steps,
            },
        }

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """Determine if the solution provided solves the GoL task.

        The function awards 1.0 for a correct answer.

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

            if correct_arr != ans_arr:
                return 0.01
            else:
                return 1.0  # Yay
        except Exception as e:
            return 0.01


register_dataset("game_of_life", GameOfLifeDataset, GameOfLifeConfig)
