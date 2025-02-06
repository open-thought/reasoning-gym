from dataclasses import dataclass
from random import Random
from typing import Dict, List, Optional, Tuple

from .contrib.sokoban.src.generator import generate
from .contrib.sokoban.src.astar import solve_astar
from .contrib.sokoban.src.game import Game
from .contrib.sokoban.src.utils import get_state, is_solved

from ..factory import ProceduralDataset, register_dataset

import numpy as np
from io import StringIO

@dataclass
class SokobanConfig:
    """Configuration for sokoban puzzle generation"""

    seed: Optional[int] = None
    size: int = 500

    # def validate(self):
    #     """Validate configuration parameters"""
    #     assert 3 <= self.grid_size_x <= 999, "grid_size_x must be between 0 and 999"
    #     assert 3 <= self.grid_size_y <= 999, "grid_size_y must be between 0 and 999"
    #     assert self.simulation_steps >= 0, "simulation_steps must be gte 0"
    #     assert self.filled_cells <= self.grid_size_x * self.grid_size_y, "filled_cells must fit in x times y"


class SokobanDataset(ProceduralDataset):
    """Generates Sokoban games with configurable parameters"""

    def __init__(self, config: SokobanConfig):
        self._prompt_templates = [
            "What will this Sokoban board look like after {simulation_steps} steps of simulation?\n\n{board}"
        ]

        super().__init__(config=config, seed=config.seed, size=config.size)

    def __getitem__(self, idx: int) -> dict:
        """Generate a single Sokoban task

        Returns:
            dict with keys:
                - question: str, the task description
                - answer: str, a solution string
                - metadata: dict with generation parameters
        """
        # rng = Random(self.seed + idx)

        # Make the Sokoban!
        (game, matrix, gamestr) = generate(seed = self.seed + idx + 1)
        
        # Solve the puzzle
        grid_list = [list(line) for line in gamestr.replace(' ', '').strip().split('\n')]
        grid_array = np.array(grid_list)
        answer = solve_astar(grid_array)

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
""" + gamestr,
            "answer": "",
            "metadata": {
                "possible_answer": answer[0],
                "gamestr": gamestr,
                "matrix": matrix
            },
        }

    def score_answer(self, answer: Optional[str], entry: Dict[str, any]) -> float:
        """Determine if the solution provided solves the Sokoban task.

        The function awards 1.0 for a correct answer.

        Args:
            answer (Optional[str]): The user's answer.
            entry (Dict[str, any]): The original dataset entry containing the correct answer.

        Returns:
            float: The computed score between 0.0 and 1.0.
        """

        if answer == None:
            return 0.0

        grid_list = [list(line) for line in entry['metadata']['gamestr'].replace(' ', '').strip().split('\n')]
        matrix = np.array(grid_list)
        state = get_state(matrix)

        game = Game()
        game.load_puzzle_matrix(matrix)
        
        for move in answer:
            game.player.update(key=move)

        if is_solved(game.get_curr_state()):
            return 1.0

        return 0.1

register_dataset("sokoban", SokobanDataset, SokobanConfig)
