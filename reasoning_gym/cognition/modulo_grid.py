from copy import deepcopy
from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from ..factory import ProceduralDataset, register_dataset


@dataclass
class ModuloGridConfig:
    """Configuration for ModuloGrid task generation"""

    size_x: int = 20
    size_y: int = 20
    max_divisor: int = 20
    max_target: int = 20
    max_holes: int = 1
    seed: Optional[int] = None
    size: int = 500

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.size_x > 5, "size_x must be greater than 5"
        assert self.size_y > 5, "size_y must be greater than 5"
        assert self.max_divisor > 0, "max_divisor must be greater than 0"
        assert self.max_target > 0, "max_target must be greater than 0"
        assert self.max_holes > 0, "max_holes must be greater than 0"


def generate_grid(size_x, size_y, operation, mod_target):
    """
    Generates a grid of symbols based on the evaluation of an operation on grid coordinates.

    Parameters:
      size_x (int): Number of columns.
      size_y (int): Number of rows.
      operation (str or callable): The operation to apply to each coordinate.
          If a string, accepted values are:
            - "sum": computes x + y.
            - "diff": computes |x - y|.
            - "prod": computes x * y.
          Otherwise, a function taking two integers (x, y) must be provided.
      mod_target (tuple): A tuple (divisor, target) such that a cell (x, y) is marked as valid
                          if (operation(x, y)) % divisor equals target.

    Returns:
      list of list: A 2D grid filled with "✅" for valid cells and "❌" for invalid cells.
    """
    # Determine the operation function
    if callable(operation):
        op_func = operation
    elif operation == "sum":
        op_func = lambda x, y: x + y
    elif operation == "diff":
        op_func = lambda x, y: abs(x - y)
    elif operation == "prod":
        op_func = lambda x, y: x * y
    elif operation == "pow":
        op_func = lambda x, y: x**y
    else:
        raise ValueError("Unsupported operation. Use 'sum', 'diff', 'prod', or provide a callable.")

    divisor, target = mod_target

    # Create the grid; using 0-indexed coordinates (x, y)
    grid = []
    for y in range(size_y):
        row = []
        for x in range(size_x):
            result = op_func(x, y)
            # Check the modulo condition
            if result % divisor == target:
                row.append("✅")
            else:
                row.append("❌")
        grid.append(row)
    return grid


def flatten_grid(grid: list[list[str]]) -> str:
    return "\n".join("".join(row) for row in grid)


class ModuloGridDataset(ProceduralDataset):
    """Generates ModuloGrid tasks

    This is an ARC-ish task for mathematical explanatory reasoning. It generates a binary grid based on a hidden
    mathematical function based around modulo division of a function based on the coordinates, then asks to fill
    in any gaps in the grid.

    The function used to determine the pattern can be based on sums, multiples, powers, and differences, then a
    constructed modulo matching a target function. Some patterns are obvious without knowing the underlying rule,
    some are very difficult. Pretty much all the parameters are configurable, so we are able to generate a
    good curriculum.
    """

    def __init__(self, config: ModuloGridConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def __getitem__(self, idx: int) -> dict:
        """Generate a single NeedleHaystack task

        Returns:
            dict with keys:
                - question: str, the task description with cube string
                - answer: None, indicating to use the dynamic evaluator
                - metadata: dict with generation parameters and example solution
        """
        rng = Random(self.seed + idx)

        valid = False
        while not valid:

            divisor = rng.randint(1, self.config.max_divisor)
            target = rng.randint(1, self.config.max_target)
            operation = rng.choice(["sum", "diff", "prod", "pow"])
            mod_target = (divisor, target)

            grid = generate_grid(self.config.size_x, self.config.size_y, operation, mod_target)
            sgrid = "".join(s for row in grid for s in row)
            if "✅" in sgrid:
                valid = True

        holes_grid = deepcopy(grid)

        for i in range(self.config.max_holes):
            holes_grid[rng.randint(0, len(holes_grid) - 1)][rng.randint(0, len(holes_grid[0]) - 1)] = "❔"

        question = (
            "Identify the mathematical pattern which defines this grid, then use that pattern to fill in the question marks. Return the entire completed grid as your answer.\n\n"
            + flatten_grid(holes_grid)
        )

        return {
            "question": question,
            "answer": flatten_grid(grid),
            "metadata": {"divisor": divisor, "target": target, "operation": operation},
        }


# Register the dataset
register_dataset("modulo_grid", ModuloGridDataset, ModuloGridConfig)
