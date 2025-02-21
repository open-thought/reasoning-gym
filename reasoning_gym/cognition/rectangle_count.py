from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from ..factory import ProceduralDataset, register_dataset

QUESTION_TEMPLATE = """Your task is to count how many rectangles are present in an ASCII grid.

Single rectangles are outlined with a '#', overlapping rectangles (max 2) are shown with '█'.

Example:
- Input: How many rectangles are in the grid below?

              ####
              #  #
              ####










 #########
 #       █##
 #       █ #
 ########█ #
         # #
         ###
- Output: 3
- Explanation:
    - The first rectangle is the 3x4 rectangle in the top right.
    - The other two rectangles are overlapping in the bottom left corner.
    - Therefore, the final answer is 3.

Now, it's your turn. How many rectangles do you see in the grid below?
{puzzle}
"""


def draw_rectangles_with_overlap(n, width, height, rng):
    # Create a grid that holds a count of how many times a cell is drawn.
    grid = [[0 for _ in range(width)] for _ in range(height)]
    rectangles = []

    max_attempts = 100000  # Prevent infinite loops in case of a crowded grid
    attempts = 0

    while len(rectangles) < n and attempts < max_attempts:
        attempts += 1
        # Ensure minimum width and height of 3.
        # For a rectangle to be at least 3 cells wide, right must be at least left + 2.
        # Similarly, bottom must be at least top + 2.
        left = rng.randint(0, width - 3)
        right = rng.randint(left + 2, width - 1)
        top = rng.randint(0, height - 3)
        bottom = rng.randint(top + 2, height - 1)

        # Prepare a list of all the cells that would be updated.
        cells_to_update = []

        # Top edge:
        for col in range(left, right + 1):
            cells_to_update.append((top, col))
        # Bottom edge:
        for col in range(left, right + 1):
            cells_to_update.append((bottom, col))
        # Left edge (excluding corners already drawn):
        for row in range(top + 1, bottom):
            cells_to_update.append((row, left))
        # Right edge (excluding corners already drawn):
        for row in range(top + 1, bottom):
            cells_to_update.append((row, right))

        # Check if drawing this rectangle would cause any cell to exceed a count of 2.
        conflict = False
        for r, c in cells_to_update:
            if grid[r][c] >= 2:
                conflict = True
                break
        if conflict:
            continue  # Skip this rectangle candidate

        # No conflict: update the grid counts.
        for r, c in cells_to_update:
            grid[r][c] += 1

        # Save the rectangle (stored as (left, right, top, bottom)).
        rectangles.append((left, right, top, bottom))

    if len(rectangles) < n:
        print(f"Only placed {len(rectangles)} rectangles after {attempts} attempts.")

    # Print the grid.
    # Use ' ' for an untouched cell, '#' for a single hit, and '█' for exactly two hits.
    lines = ""
    for row in grid:
        line = "".join(" " if count == 0 else ("#" if count == 1 else "█") for count in row)
        lines = lines + line + "\n"
    return lines, len(rectangles)


@dataclass
class RectangleCountConfig:
    """Configuration for RectangleCount puzzle generation"""

    max_rectangles: int = 10
    width: int = 80
    height: int = 80
    seed: Optional[int] = None
    size: int = 500

    def validate(self):
        """Validate configuration parameters"""
        assert self.width >= 10, "width must be gte 10"
        assert self.height >= 10, "height must be gte 10"


class RectangleCountDataset(ProceduralDataset):
    """Generates ASCII rectangle counting puzzles with configurable parameters"""

    def __init__(self, config: RectangleCountConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def __getitem__(self, idx: int) -> dict:
        """Generate a single RectangleCount task

        Returns:
            dict with keys:
                - question: str, the task description
                - answer: str, a solution string
                - metadata: dict with generation parameters
        """
        rng = Random(self.seed + idx)

        target = rng.randint(1, self.config.max_rectangles)
        puzzle, answer = draw_rectangles_with_overlap(target, self.config.width, self.config.height, rng)

        return {
            "question": QUESTION_TEMPLATE.format(puzzle=puzzle),
            "answer": str(answer),
            "metadata": {"puzzle": puzzle, "solution": answer},
        }

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """Determine if the solution provided solves the RectangleCount task.

        The function awards 1.0 for a correct answer.

        Args:
            answer (Optional[str]): The user's answer.
            entry (dict[str, Any]): The original dataset entry containing the correct answer.

        Returns:
            float: The computed score between 0.0 and 1.0.
        """

        if answer == None:
            return 0.0
        if answer.lower().replace("\n", "") != entry["answer"].lower().replace("\n", ""):
            return 0.01
        else:
            return 1.0  # Yay


register_dataset("rectangle_count", RectangleCountDataset, RectangleCountConfig)
