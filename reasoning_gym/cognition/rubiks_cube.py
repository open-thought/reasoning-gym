import re
from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from magiccube.cube import Cube, CubeMove, CubeMoveType
from magiccube.solver.basic.basic_solver import BasicSolver

from ..factory import ProceduralDataset, register_dataset


@dataclass
class RubiksCubeConfig:
    """Configuration for RubiksCube task generation"""

    scramble_steps: int = 3  # Number of random steps from initial state
    cube_size: int = 3  # Default to a standard 3x3x3 cube
    remove_ansi: bool = True
    seed: Optional[int] = None
    size: int = 500

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
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _generate_random_moves(self, rng: Random, cube: Cube, num_steps: int = 50, wide=None) -> list[CubeMove]:
        """Generate a list of random moves (but don't apply them).
        By default scramble only uses wide moves to cubes with size >=4."""

        if wide is None and cube.size <= 3:
            wide = False
        elif wide is None and cube.size > 3:
            wide = True

        possible_moves = [
            CubeMoveType.L,
            CubeMoveType.R,  # CubeMoveType.M,
            CubeMoveType.D,
            CubeMoveType.U,  # CubeMoveType.E,
            CubeMoveType.B,
            CubeMoveType.F,  # CubeMoveType.S,
        ]
        movements = [
            CubeMove(
                rng.choice(possible_moves),
                rng.choice([False, True]),  # reversed
                rng.choice([False, True]) if wide else False,  # wide
                rng.randint(1, cube.size // 2) if wide else 1,  # layer
            )
            for _ in range(num_steps)
        ]

        return movements

    def __getitem__(self, idx: int) -> dict:
        """Generate a single RubiksCube task

        Returns:
            dict with keys:
                - question: str, the task description with cube string
                - answer: None, indicating to use the dynamic evaluator
                - metadata: dict with generation parameters and example solution
        """
        rng = Random(self.seed + idx)

        cube = Cube(self.config.cube_size)
        scramble_moves = self._generate_random_moves(rng, cube, num_steps=self.config.scramble_steps)
        cube.rotate(scramble_moves)

        # render cube
        if self.config.remove_ansi:
            cube_render = self.remove_ansi(str(cube))
        else:
            cube_render = str(cube)

        if self.config.cube_size == 3:
            solver = BasicSolver(cube)
            actions = solver.solve()
            actions_string = " ".join([str(move) for move in actions])
        else:
            actions = None

        return {
            "question": rng.choice(self._prompt_templates).format(
                cube_size=self.config.cube_size, cube_render=cube_render
            ),
            "answer": None,
            "metadata": {
                "cube_size": self.config.cube_size,
                "scramble_steps": self.config.scramble_steps,
                "scramble_moves": " ".join([str(move) for move in scramble_moves]),
                "example_correct_answer": actions_string,
            },
        }

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """Determine if the solution provided solves the cube"""
        reward = 0.0  # default reward
        if answer is not None:
            # Reconstruct the test cube
            eval_cube = Cube(entry["metadata"]["cube_size"])
            eval_cube.rotate(entry["metadata"]["scramble_moves"])

            # Test the solution
            try:
                eval_cube.rotate(answer)
                solved = eval_cube.is_done()

                if solved:
                    reward = 1.0
                elif len(answer.strip()) > 0:  # encourage non-empty answers
                    reward = 0.05  # Incorrect, but rotate could parse the answer
                else:
                    reward = 0.01
            except:
                reward = 0.01  # At least you tried

        return reward

    def remove_ansi(self, line):
        """Remove terminal colors from magiccube rendering"""
        ansi_escape = re.compile(r"(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]")
        return ansi_escape.sub("", line)


# Register the dataset
register_dataset("rubiks_cube", RubiksCubeDataset, RubiksCubeConfig)
