import random
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Optional

from ..factory import ProceduralDataset, register_dataset


class Color(StrEnum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    YELLOW = "yellow"
    WHITE = "white"
    ORANGE = "orange"
    PURPLE = "purple"
    PINK = "pink"
    BROWN = "brown"
    GRAY = "gray"
    CYAN = "cyan"
    MAGENTA = "magenta"
    GOLD = "gold"
    SILVER = "silver"
    INDIGO = "indigo"
    VIOLET = "violet"


class Side(StrEnum):
    TOP = "top"
    RIGHT = "right"
    FRONT = "front"
    LEFT = "left"
    BACK = "back"
    BOTTOM = "bottom"


@dataclass
class Cube:
    """Represents a cube with colored sides"""

    colors: dict[Side, Color]

    def rotate_front_to_top(self) -> None:
        """Rotate cube so front face becomes top"""
        old = self.colors.copy()
        self.colors[Side.TOP] = old[Side.FRONT]
        self.colors[Side.FRONT] = old[Side.BOTTOM]
        self.colors[Side.BOTTOM] = old[Side.BACK]
        self.colors[Side.BACK] = old[Side.TOP]
        # Right and left stay in place

    def rotate_right_to_top(self) -> None:
        """Rotate cube so right face becomes top"""
        old = self.colors.copy()
        self.colors[Side.TOP] = old[Side.RIGHT]
        self.colors[Side.RIGHT] = old[Side.BOTTOM]
        self.colors[Side.BOTTOM] = old[Side.LEFT]
        self.colors[Side.LEFT] = old[Side.TOP]
        # Front and back stay in place

    def rotate_back_to_top(self) -> None:
        """Rotate cube so back face becomes top"""
        old = self.colors.copy()
        self.colors[Side.TOP] = old[Side.BACK]
        self.colors[Side.BACK] = old[Side.BOTTOM]
        self.colors[Side.BOTTOM] = old[Side.FRONT]
        self.colors[Side.FRONT] = old[Side.TOP]
        # Right and left stay in place

    def rotate_left_to_top(self) -> None:
        """Rotate cube so left face becomes top"""
        old = self.colors.copy()
        self.colors[Side.TOP] = old[Side.LEFT]
        self.colors[Side.LEFT] = old[Side.BOTTOM]
        self.colors[Side.BOTTOM] = old[Side.RIGHT]
        self.colors[Side.RIGHT] = old[Side.TOP]
        # Front and back stay in place

    def rotate_bottom_to_top(self) -> None:
        """Rotate cube so bottom face becomes top"""
        old = self.colors.copy()
        self.colors[Side.TOP] = old[Side.BOTTOM]
        self.colors[Side.BOTTOM] = old[Side.TOP]
        self.colors[Side.FRONT] = old[Side.BACK]
        self.colors[Side.BACK] = old[Side.FRONT]
        # Right and left stay in place


@dataclass
class ColorCubeRotationConfig:
    """Configuration for color cube rotation task generation"""

    min_rotations: int = 1
    max_rotations: int = 3
    seed: Optional[int] = None
    size: int = 500

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.min_rotations > 0, "min_rotations must be positive"
        assert self.max_rotations >= self.min_rotations, "max_rotations must be >= min_rotations"


class ColorCubeRotationDataset(ProceduralDataset):
    """Generates color cube rotation reasoning tasks"""

    def __init__(self, config: ColorCubeRotationConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def __getitem__(self, idx: int) -> dict:
        rng = random.Random(self.seed + idx)

        # Generate initial cube state
        cube = self._generate_cube(rng)
        initial_state = cube.colors.copy()

        # Generate sequence of rotations
        num_rotations = rng.randint(self.config.min_rotations, self.config.max_rotations)
        rotations = []

        # Keep trying until we have at least one valid rotation
        while len(rotations) < num_rotations:
            # Get all sides except TOP
            available_sides = [s for s in Side if s != Side.TOP]
            from_side = rng.choice(available_sides)
            rotations.append(from_side)
            self._rotate_to_top(cube, from_side)

        # Select target side for question
        target_side = rng.choice(list(Side))

        # Generate story
        story = self._generate_story(initial_state, rotations, target_side, rng)

        return {
            "question": story,
            "answer": cube.colors[target_side],
            "metadata": {
                "initial_state": {k.value: v.value for k, v in initial_state.items()},
                "rotations": [r.value for r in rotations],
                "target_side": target_side.value,
                "num_rotations": num_rotations,
            },
        }

    def _generate_cube(self, rng: random.Random) -> Cube:
        """Generate a cube with random colors"""
        colors = list(Color)
        rng.shuffle(colors)  # Randomize color order
        return Cube({side: color for side, color in zip(Side, colors)})

    def _rotate_to_top(self, cube: Cube, from_side: Side) -> None:
        """Rotate cube so that given side becomes top"""
        rotation_map = {
            Side.FRONT: cube.rotate_front_to_top,
            Side.RIGHT: cube.rotate_right_to_top,
            Side.BACK: cube.rotate_back_to_top,
            Side.LEFT: cube.rotate_left_to_top,
            Side.BOTTOM: cube.rotate_bottom_to_top,
        }
        if from_side in rotation_map:
            rotation_map[from_side]()

    def _generate_story(
        self, initial_state: dict[Side, Color], rotations: list[Side], target_side: Side, rng: random.Random
    ) -> str:
        """Generate story describing cube state and rotations"""
        # Describe initial state
        story_parts = ["A cube has:"]
        for side in Side:
            story_parts.append(f"- a {initial_state[side].value} {side.value} side")

        # Describe rotations
        rotation_templates = [
            "The cube is rotated so that the side which was before at the {side} is now at the top.",
            "Then the cube is rotated to bring the {side} side to the top.",
            "After that the cube is turned to make the {side} face the top.",
            "Now the cube is rotated to place its {side} side at the top.",
            "Next, the {side} side is rotated to become the top face.",
        ]

        for i, from_side in enumerate(rotations):
            template = rotation_templates[0] if i == 0 else rng.choice(rotation_templates[1:])
            story_parts.append(f"\n{template.format(side=from_side.value)}")

        # Ask question
        story_parts.append(f"\nWhat is now the color of the {target_side.value} side of the cube?")
        story_parts.append(f"Provide only the color as your final answer.")

        return "\n".join(story_parts)

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        reward = 0.0
        metadata = entry["metadata"]
        if answer is not None:
            try:
                answer_formatted = answer.lower()
                solved = answer_formatted == metadata["answer"]
                if solved:
                    reward = 1.0
                elif metadata["answer"] in answer_formatted:
                    reward = 0.25
                elif len(answer.strip()) > 0:
                    reward = 0.05
                else:
                    reward = 0.01
            except:
                reward = 0.01
        return reward


register_dataset("color_cube_rotation", ColorCubeRotationDataset, ColorCubeRotationConfig)
