import re
from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from magiccube.cube import Cube, CubeMove, CubeMoveType
from magiccube.solver.basic.basic_solver import BasicSolver

from ..coaching import BaseCurriculum, RangeAttributeDefinition, ScalarAttributeDefinition
from ..factory import ProceduralDataset, register_dataset

DATASET_NAME = "rubiks_cube"


@dataclass
class RubiksCubeConfig:
    """Configuration for RubiksCube task generation"""

    min_scramble_steps: int = 3  # Minimum number of random steps from initial state
    max_scramble_steps: int = 10  # Maximum number of random steps from initial state
    cube_size: int = 3  # Default to a standard 3x3x3 cube
    remove_ansi: bool = True
    seed: Optional[int] = None
    size: int = 500

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.cube_size > 1, "cube_size must be greater than 1"
        assert self.cube_size < 7, "cube_size must be less than 7"
        assert self.min_scramble_steps > 0, "min_scramble_steps must be greater than 0"
        assert (
            self.max_scramble_steps >= self.min_scramble_steps
        ), "max_scramble_steps must be greater than min_scramble_steps"


class RubiksCubeDataset(ProceduralDataset):
    """Generates RubiksCube tasks"""

    def __init__(self, config: RubiksCubeConfig):
        self._prompt_templates = [
            "You are given a {cube_size}x{cube_size}x{cube_size} Rubik's cube. It looks like this:\n\n{cube_render} \n\nPlease provide a solution to solve this cube using Singmaster notation. Do not combine any steps, for instance, do not write 'U2', and instead write 'U U'.",
            "You see a size {cube_size} Rubik's cube arranged as follows::\n\n{cube_render} \n\nPlease provide a solution to solve this cube using Singmaster notation. Do not combine any steps, for instance, do not write 'U2', and instead write 'U U'.",
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
        num_steps = rng.randint(self.config.min_scramble_steps, self.config.max_scramble_steps)
        scramble_moves = self._generate_random_moves(rng, cube, num_steps=num_steps)
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
            actions_string = ""

        return {
            "question": rng.choice(self._prompt_templates).format(
                cube_size=self.config.cube_size, cube_render=cube_render
            ),
            "answer": None,
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "cube_size": self.config.cube_size,
                "scramble_steps": num_steps,
                "scramble_moves": " ".join([str(move) for move in scramble_moves]),
                "example_correct_answer": actions_string,
                "difficulty": {
                    "cube_size": self.config.cube_size,
                    "scramble_steps": (self.config.min_scramble_steps, self.config.max_scramble_steps),
                },
            },
        }

    def partial_score(self, cube: Cube) -> float:
        """
        Returns a fraction between 0 and 1, indicating how many stickers are
        correctly positioned (i.e., match the solved color for that face).
        """
        total_stickers = 6 * (cube.size**2)
        correct_stickers = 0

        for face_index in range(6):
            face = cube.faces[face_index]

            solved_color = face[cube.size // 2][cube.size // 2].color
            for row in range(cube.size):
                for col in range(cube.size):
                    sticker = face[row][col]
                    if sticker.color == solved_color:
                        correct_stickers += 1

        return correct_stickers / total_stickers

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """Determine if the solution provided solves the cube, with partial rewards."""
        reward = 0.0  # default
        if answer is not None:
            eval_cube = Cube(entry["metadata"]["cube_size"])
            eval_cube.rotate(entry["metadata"]["scramble_moves"])
            try:
                expanded_answer = self.expand_moves(answer)
                eval_cube.rotate(expanded_answer)

                # 3) Check if fully solved
                solved = eval_cube.is_done()
                if solved:
                    reward = 1.0
                else:
                    partial = self.partial_score(eval_cube)

                    if len(answer.strip()) > 0:
                        reward = max(0.05, partial)
                    else:
                        reward = max(0.01, partial)
            except:
                reward = 0.01
        return reward

    def remove_ansi(self, line):
        """Remove terminal colors from magiccube rendering"""
        ansi_escape = re.compile(r"(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]")
        return ansi_escape.sub("", line)

    def expand_moves(self, move_str):
        try:
            moves = move_str.split()
            expanded = []
            for move in moves:
                # Split the move into the base part and any trailing digits
                match = re.fullmatch(r"^([^\d]*)(\d*)$", move)
                if match:
                    base, num_part = match.groups()
                    if num_part:
                        expanded.extend([base] * int(num_part))
                    else:
                        expanded.append(base)
            return " ".join(expanded).strip()
        except Exception as e:
            return move_str


class RubiksCubeCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(RubiksCubeCurriculum.__name__, RubiksCubeConfig)

        # Define attributes
        self._define_attributes(
            ScalarAttributeDefinition(
                name="cube_size",
                field_name="cube_size",
                levels=[3, 4, 5, 6, 7],
                description="Board size",
            ),
            RangeAttributeDefinition(
                name="scramble_steps",
                levels=[3, 10, 25, 50, 100],
                description="Number of random moves to scramble the cube",
                lower_field_name="min_scramble_steps",
                upper_field_name="max_scramble_steps",
                ensure_interval=True,
            ),
        )


# Register the dataset
register_dataset(DATASET_NAME, RubiksCubeDataset, RubiksCubeConfig, RubiksCubeCurriculum)
