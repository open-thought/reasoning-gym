from dataclasses import dataclass
from random import Random
from typing import Any, Optional

import numpy as np

from ..factory import ProceduralDataset, register_dataset


@dataclass
class SokobanConfig:
    """Configuration for sokoban puzzle generation"""

    min_w: int = 6  # Minimum width of the puzzle
    min_h: int = 6  # Minimum height of the puzzle
    max_w: int = 10  # Maximum width of the puzzle
    max_h: int = 10  # Maximum height of the puzzle
    min_boxes: int = 4  # Minimum number of boxes
    max_boxes: int = 10  # Maximum number of boxes
    max_depth: int = 80  # Maximum search depth
    seed: Optional[int] = None
    size: int = 500

    def validate(self):
        """Validate configuration parameters"""
        assert 0 < self.max_w <= 20
        assert 0 < self.max_h <= 20
        assert self.min_h > 0
        assert self.min_w > 0
        assert self.min_w <= self.max_w, "min_w must be lte max_w"
        assert self.min_h <= self.max_h, "min_h must be lte max_h"
        assert self.min_boxes <= self.max_boxes, "min_boxes must be lte max_boxes"
        assert self.max_depth > 1


class SokobanDataset(ProceduralDataset):
    """Generates Sokoban games with configurable parameters"""

    def __init__(self, config: SokobanConfig):
        self._prompt_templates = [
            "What will this Sokoban board look like after {simulation_steps} steps of simulation?\n\n{board}"
        ]

        super().__init__(config=config, seed=config.seed, size=config.size)

        # lazy loading of sokoban imports
        from .contrib.sokoban.src.game import Game
        from .contrib.sokoban.src.generator import generate
        from .contrib.sokoban.src.utils import is_solved

        self._Game = Game
        self._generate = generate
        self._is_solved = is_solved

    def __getitem__(self, idx: int) -> dict:
        """Generate a single Sokoban task

        Returns:
            dict with keys:
                - question: str, the task description
                - answer: str, a solution string
                - metadata: dict with generation parameters
        """

        # Make the Sokoban!
        rng = Random(self.seed + idx)
        gamestr, solution, difficulty = self._generate(
            rng=rng,
            min_w=self.config.min_w,
            min_h=self.config.min_h,
            max_w=self.config.max_w,
            max_h=self.config.max_h,
            min_boxes=self.config.min_boxes,
            max_boxes=self.config.max_boxes,
            max_depth=self.config.max_depth,
        )

        return {
            "question": """You are going to solve a 'sokoban' puzzle.

* - The player
% - The player on a goal
@ - A box
X - A goal
$ - A box on a goal
+ - A wall
- - An empty position

Your solution must be a string of characters, ex: LDURRUDL.

Here is your puzzle:
"""
            + gamestr,
            "answer": solution,
            "metadata": {"gamestr": gamestr, "difficulty": difficulty},
        }

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """Determine if the solution provided solves the Sokoban task.

        The function awards 1.0 for a correct answer.

        Args:
            answer (Optional[str]): The user's answer.
            entry (dict[str, Any]): The original dataset entry containing the correct answer.

        Returns:
            float: The computed score between 0.0 and 1.0.
        """

        if not isinstance(answer, str):
            return 0.0

        try:
            grid_list = [list(line) for line in entry["metadata"]["gamestr"].replace(" ", "").strip().split("\n")]
            matrix = np.array(grid_list)

            h, w = matrix.shape
            game = self._Game(height=h, width=w)
            game.load_puzzle_matrix(matrix)

            for move in answer:
                game.player.update(key=move)

            if self._is_solved(game.get_curr_state()):
                return 1.0
        except:
            pass

        return 0.0


register_dataset("sokoban", SokobanDataset, SokobanConfig)
