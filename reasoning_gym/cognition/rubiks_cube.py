from dataclasses import dataclass
import random
import re

from magiccube.cube import Cube
from magiccube.solver.basic.basic_solver import BasicSolver

from typing import List, Optional, Tuple, Dict


from ..factory import ProceduralDataset, register_dataset


@dataclass
class RubiksCubeConfig:
    """Configuration for RubiksCube task generation"""

    scramble_steps: int = 3  # Number of random steps from initial state
    cube_size: int = 3  # Default to a standard 3x3x3 cube

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.cube_size > 1, "cube_size must be greater than 1"
        assert self.cube_size < 7, "cube_size must be less than 7"
        assert self.scramble_steps > 0, "scramble_steps must be greater than 0"

class RubiksCubeDataset(ProceduralDataset):
    """Generates RubiksCube tasks"""

    def __init__(self, config: RubiksCubeConfig):
        self._prompt_templates = [
            "You are given a {cube_size}x{cube_size}x{cube_size} Rubik's cube. It looks like this:\n\n{cube_render} \n\nPlease provide a solution to solve this cube using Singmaster notation.",
            "You see a size {cube_size} Rubik's cube. It is arranged this:\n\n{cube_render} \n\nPlease provide a solution to solve this cube.",
        ]
        super().__init__(config=config)

    def __getitem__(self, idx: int) -> dict:
        """Generate a single RubiksCube task

        Returns:
            dict with keys:
                - question: str, the task description with cube string
                - answer: None, indicating to use the dynamic evaluator
                - metadata: dict with generation parameters and example solution
        """

        cube = Cube(self.config.cube_size)
        scramble_moves = cube.scramble(num_steps=self.config.scramble_steps)
        cube_render = self.remove_ansi(str(cube))

        if self.config.cube_size == 3:
            solver = BasicSolver(cube)
            actions = solver.solve()
            actions_string = ' '.join([str(move) for move in actions])
        else:
            actions = None

        return {
            "question": random.choice(self._prompt_templates).format(cube_size=self.config.cube_size, cube_render=cube_render),
            "answer": None,
            "metadata": {
                "cube_size": self.config.cube_size,
                "scramble_steps": self.config.scramble_steps,
                "scramble_moves": ' '.join([str(move) for move in scramble_moves]),
                "example_correct_answer": actions_string,
            },
        }

    def score_answer(self, answer: Optional[str], entry: Dict[str, any]) -> float:
        """ Determine if the solution provided solves the cube """

        reward = 0.0
        if answer is not None:

            # Reconstruct the test cube
            eval_cube = Cube(entry["metadata"]["cube_size"])
            eval_cube.rotate(entry["metadata"]["scramble_moves"])
            
            # Test the solution
            eval_cube.rotate(answer)
            solved = eval_cube.is_done()

            if solved:
                reward = 1.0
            else:
                reward = 0.01 # At least you tried

        return reward

    def remove_ansi(self, line):
        """ Remove terminal colors from magiccube rendering"""
        ansi_escape = re.compile(r'(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]')
        return ansi_escape.sub('', line)


# Register the dataset
register_dataset("RubiksCube", RubiksCubeDataset, RubiksCubeConfig)
