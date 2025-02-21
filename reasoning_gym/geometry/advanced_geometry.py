import random
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import sympy
from sympy.geometry import Point

from ..factory import ProceduralDataset, register_dataset


@dataclass
class AdvancedGeometryConfig:
    """
    Configuration for generating advanced geometry tasks.
    """

    min_coord: int = -10  # Minimum x/y coordinate
    max_coord: int = 10  # Maximum x/y coordinate
    size: int = 50  # Number of problems to generate
    seed: Optional[int] = None

    # Probability or list of tasks we want to generate
    # For demonstration, we have three categories:
    task_types: list[str] = field(
        default_factory=lambda: [
            "orthocenter",
            "incircle_radius",
            "angle_measure",
        ]
    )

    def validate(self):
        assert self.min_coord < self.max_coord, "min_coord must be < max_coord."
        assert self.size > 0, "Size of dataset must be positive."
        assert len(self.task_types) > 0, "Must specify at least one task type."


# Join format instructions into a single string
GEOMETRY_FORMAT_INSTRUCTIONS = "\n".join(
    [
        "For all geometry problems:",
        "1. Give coordinates in the form (x, y)",
        "2. Round decimal answers to 3 decimal places",
        "3. Use the degree symbol ° for angles",
        "4. Return only th angle, coordinates, or radius as your answer.",
    ]
)


class AdvancedGeometryDataset(ProceduralDataset):
    """
    A dataset for advanced geometry tasks using coordinate geometry.
    """

    def __init__(self, config: AdvancedGeometryConfig):
        self._prompt_templates = {
            "orthocenter": [
                f"Given triangle ABC with coordinates A={{A}}, B={{B}}, and C={{C}}, find the coordinates of its orthocenter. {GEOMETRY_FORMAT_INSTRUCTIONS}",
                f"For triangle with vertices A={{A}}, B={{B}}, and C={{C}}, determine the orthocenter (intersection of altitudes). {GEOMETRY_FORMAT_INSTRUCTIONS}",
            ],
            "incircle_radius": [
                f"Consider triangle ABC with coordinates A={{A}}, B={{B}}, and C={{C}}. Compute the radius of its incircle. {GEOMETRY_FORMAT_INSTRUCTIONS}",
                f"Find the incircle radius of triangle ABC whose vertices are A={{A}}, B={{B}}, and C={{C}}. {GEOMETRY_FORMAT_INSTRUCTIONS}",
            ],
            "angle_measure": [
                f"In triangle ABC with coordinates A={{A}}, B={{B}}, and C={{C}}, find the measure (in degrees) of angle ABC. {GEOMETRY_FORMAT_INSTRUCTIONS}",
                f"Given a triangle with vertices A={{A}}, B={{B}}, and C={{C}}, determine the angle at B in degrees. {GEOMETRY_FORMAT_INSTRUCTIONS}",
            ],
        }
        super().__init__(config=config, seed=config.seed, size=config.size)

    def __getitem__(self, idx: int) -> dict:
        """
        Generate a single advanced geometry item based on the config's task types.
        """
        rng = random.Random(self.seed + idx)
        task_type = rng.choice(self.config.task_types)

        # Randomly generate coordinates for a triangle
        A, B, C = self._generate_non_degenerate_triangle(rng)

        # Build a question and compute the solution
        if task_type == "orthocenter":
            question, answer, metadata = self._build_orthocenter_task(rng, A, B, C)
        elif task_type == "incircle_radius":
            question, answer, metadata = self._build_incircle_radius_task(rng, A, B, C)
        elif task_type == "angle_measure":
            question, answer, metadata = self._build_angle_measure_task(rng, A, B, C)
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

        metadata["task_type"] = task_type

        return {
            "question": question,
            "answer": answer,
            "metadata": metadata,
        }

    def _generate_non_degenerate_triangle(self, rng: random.Random):
        """
        Generate a random non-degenerate triangle with integer coordinates
        in [min_coord, max_coord] x [min_coord, max_coord].
        """
        max_attempts = 100
        for _ in range(max_attempts):
            # Generate points with integer coordinates
            points = []
            for _ in range(3):
                x = rng.randint(self.config.min_coord, self.config.max_coord)
                y = rng.randint(self.config.min_coord, self.config.max_coord)
                points.append(Point(x, y))

            A, B, C = points

            # Calculate signed area to check for non-degeneracy
            # Using the formula: 1/2 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
            area = abs(A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y)) / 2

            if area > 0:
                return A, B, C

        raise ValueError(f"Failed to generate a non-degenerate triangle after {max_attempts} attempts.")

    def _build_orthocenter_task(self, rng: random.Random, A: Point, B: Point, C: Point):
        """
        Build a question about finding the orthocenter of triangle ABC.
        """
        # Convert segments to lines
        BC_line = sympy.Line(B, C)
        CA_line = sympy.Line(C, A)

        # Calculate altitudes by creating lines perpendicular from each vertex
        alt_A = BC_line.perpendicular_line(A)
        alt_B = CA_line.perpendicular_line(B)

        # Find orthocenter (intersection of any two altitudes, e.g. alt_A and alt_B)
        ortho = alt_A.intersection(alt_B)[0]

        x_ortho_approx = float(ortho.x.evalf())
        y_ortho_approx = float(ortho.y.evalf())

        question_template = rng.choice(self._prompt_templates["orthocenter"])
        question = question_template.format(A=(A.x, A.y), B=(B.x, B.y), C=(C.x, C.y), a="a", b="b")
        answer_str = f"({x_ortho_approx:.3f}, {y_ortho_approx:.3f})"

        metadata = {
            "A": (A.x, A.y),
            "B": (B.x, B.y),
            "C": (C.x, C.y),
            "ortho": (ortho.x, ortho.y),
            "orthocenter_exact": (str(ortho.x), str(ortho.y)),
            "orthocenter_approx": (x_ortho_approx, y_ortho_approx),
        }
        return question, answer_str, metadata

    def _build_incircle_radius_task(self, rng: random.Random, A: Point, B: Point, C: Point):
        """
        Build a question about finding the incircle radius of triangle ABC.
        """
        # Calculate side lengths
        a = B.distance(C)
        b = C.distance(A)
        c = A.distance(B)

        # Semi-perimeter
        s = (a + b + c) / 2

        # Area using Heron's formula
        area = sympy.sqrt(s * (s - a) * (s - b) * (s - c))

        # Radius of incircle = Area / Semi-perimeter
        radius = area / s

        # Convert to float for final answer
        radius_approx = float(radius.evalf())

        question_template = rng.choice(self._prompt_templates["incircle_radius"])
        question = question_template.format(A=(A.x, A.y), B=(B.x, B.y), C=(C.x, C.y))
        answer_str = f"{radius_approx:.3f}"

        metadata = {
            "A": (A.x, A.y),
            "B": (B.x, B.y),
            "C": (C.x, C.y),
            "incircle_radius_exact": str(radius),
            "incircle_radius_approx": radius_approx,
        }
        return question, answer_str, metadata

    def _build_angle_measure_task(self, rng: random.Random, A: Point, B: Point, C: Point):
        """
        Build a question about finding the measure of angle ABC in degrees.
        """
        # Angle at B means the angle ∠ABC
        # Vector BA = A - B, BC = C - B
        BA = A - B
        BC = C - B

        # Use vector dot product to find angle between BA and BC
        # angle = arccos((BA · BC) / (|BA| * |BC|))
        dot_val = BA.dot(BC)
        mag_ba = BA.distance(Point(0, 0))
        mag_bc = BC.distance(Point(0, 0))

        # numerical check
        if mag_ba == 0 or mag_bc == 0:
            # degenerate, but theoretically we forced a non-degenerate triangle
            angle_deg = 0
        else:
            cos_theta = dot_val / (mag_ba * mag_bc)
            # clamp cos_theta to [-1, 1] to avoid floating rounding errors
            cos_theta = max(-1, min(1, cos_theta))
            angle_rad = sympy.acos(cos_theta)
            angle_deg = float(angle_rad.evalf() * 180 / sympy.pi)

        question_template = rng.choice(self._prompt_templates["angle_measure"])
        question = question_template.format(A=(A.x, A.y), B=(B.x, B.y), C=(C.x, C.y), a="a", b="b")

        answer_str = f"{angle_deg:.2f}°"
        metadata = {
            "A": (A.x, A.y),
            "B": (B.x, B.y),
            "C": (C.x, C.y),
            "angle_ABC_degrees": angle_deg,
        }
        return question, answer_str, metadata

    def score_answer(self, answer: str | None, entry: dict[str, Any]) -> float:
        reward = 0.0
        expected_answer = entry["answer"]
        metadata = entry["metadata"]
        task_type = metadata["task_type"]

        if answer is not None:
            try:
                if metadata["task_type"] == "angle_measure":
                    answer = answer.replace("\u00b0", "")
                    expected_answer = expected_answer.replace("\u00b0", "")
                    if np.round(float(answer), 2) == np.round(float(expected_answer), 2):
                        reward = 1.0
                    elif len(answer.strip()) > 0:
                        reward = 0.05
                    else:
                        reward = 0.01
                elif metadata["task_type"] == "orthocenter":
                    x_coord = answer.split(",")[0].replace("(", "").strip()
                    y_coord = answer.split(",")[1].replace(")", "").strip()

                    expected_x = metadata["ortho"][0]
                    expected_y = metadata["ortho"][1]

                    if x_coord == expected_x and y_coord == expected_y:
                        reward = 1.0
                    elif (np.round(float(x_coord), 2) == np.round(float(expected_x), 2)) and (
                        np.round(float(y_coord), 2) == np.round(float(expected_y), 2)
                    ):
                        reward = 1.0
                    elif len(x_coord.strip()) > 0 and len(y_coord.strip()) > 0:
                        reward = 0.05
                    else:
                        reward = 0.01
                elif metadata["task_type"] == "incircle_radius":
                    if answer == expected_answer:
                        reward = 1.0
                    elif np.round(float(answer), 2) == np.round(float(metadata["incircle_radius_exact"]), 2):
                        reward = 0.5
                    elif len(answer.strip()) > 0:
                        reward = 0.05
                    else:
                        reward = 0.01
                else:
                    raise ValueError(f"Unknown task_type: {task_type}")
            except:
                reward = 0.01
        return reward


# Register the dataset
register_dataset("advanced_geometry", AdvancedGeometryDataset, AdvancedGeometryConfig)
