from dataclasses import dataclass
from random import Random
from typing import Dict, List, Optional

import cellpylib as cpl

from ..factory import ProceduralDataset, register_dataset


@dataclass
class GameOfLifeHaltingConfig:
    """Configuration for Game of Life halting problems generation"""

    grid_size_x: int = 12
    grid_size_y: int = 12
    difficulty: int = 1
    num_oscillators: int = 5
    max_simulation_steps: int = 20
    seed: Optional[int] = None
    size: int = 500

    def validate(self):
        """Validate configuration parameters"""
        assert self.difficulty in (1, 2, 3), "difficulty must be one of (1, 2, 3)"
        if self.difficulty == 1:
            assert self.grid_size_x >= 7, "grid_size_x must be gte 7 (difficulty 1)"
            assert self.grid_size_y >= 7, "grid_size_y must be gte 7 (difficulty 1)"
        if self.difficulty == 2:
            assert self.grid_size_x >= 13, "grid_size_x must be gte 13 (difficulty 2)"
            assert self.grid_size_y >= 13, "grid_size_y must be gte 13 (difficulty 2)"
        if self.difficulty == 3:
            assert self.grid_size_x >= 25, "grid_size_x must be gte 25 (difficulty 3)"
            assert self.grid_size_y >= 25, "grid_size_y must be gte 25 (difficulty 3)"


class GameOfLifeHaltingDataset(ProceduralDataset):
    """Generates Game of Life games with configurable parameters

    This is a variant of the Game of Life task, which rather than trying to test the algorithmic simulation, tests
    the ability of the model to do explanatory reasoning of the board. The idea is that a model with good
    explanatory reasoning will be able to see that a game will not halt without simulating it into the future.

    The task presents a GoL board, and the model is asked to predict if the board will halt (die, all cells zero)
    after n steps. Sometimes, the board will be made up of 'oscillators', isolated structures which never die.
    Othertimes, it is filled with non-oscillators, structures which will always die after a few steps. The model
    should deduce which case the presented board is.
    """

    # via this great wiki https://conwaylife.com/wiki/oscillator
    OSCILLATORS = [
        # Easy
        {
            "name": "blinker",
            "size_x": 3,
            "size_y": 3,
            "period": 2,
            "difficulty": 1,
            "cells": [
                [0, 0, 0],
                [1, 1, 1],
                [0, 0, 0],
            ],
        },
        {
            "name": "toad",
            "size_x": 4,
            "size_y": 4,
            "period": 2,
            "difficulty": 1,
            "cells": [
                [0, 1, 1, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 1, 1, 0],
            ],
        },
        {
            "name": "clock",
            "size_x": 4,
            "size_y": 4,
            "period": 2,
            "difficulty": 1,
            "cells": [
                [0, 0, 1, 0],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 1, 0, 0],
            ],
        },
        {
            "name": "bipole",
            "size_x": 5,
            "size_y": 5,
            "period": 2,
            "difficulty": 1,
            "cells": [
                [0, 0, 0, 1, 1],
                [0, 0, 1, 0, 1],
                [0, 0, 0, 0, 0],
                [1, 0, 1, 0, 0],
                [1, 1, 0, 0, 0],
            ],
        },
        {
            "name": "tripole",
            "size_x": 6,
            "size_y": 6,
            "period": 2,
            "difficulty": 1,
            "cells": [
                [0, 0, 0, 0, 1, 1],
                [0, 0, 0, 1, 0, 1],
                [0, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
            ],
        },
        # Medium
        {
            "name": "caterer",
            "size_x": 6,
            "size_y": 9,
            "period": 3,
            "difficulty": 2,
            "cells": [
                [0, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0],
                [0, 1, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ],
        },
        {
            "name": "mold",
            "size_x": 6,
            "size_y": 6,
            "period": 4,
            "difficulty": 2,
            "cells": [
                [0, 0, 0, 1, 1, 0],
                [0, 0, 1, 0, 0, 1],
                [1, 0, 0, 1, 0, 1],
                [0, 0, 0, 0, 1, 0],
                [1, 0, 1, 1, 0, 0],
                [0, 1, 0, 0, 0, 0],
            ],
        },
        {
            "name": "pinwheel",
            "size_x": 12,
            "size_y": 12,
            "period": 4,
            "difficulty": 2,
            "cells": [
                [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
                [1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1],
                [0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            ],
        },
        # Hard
        {
            "name": "pentadecathlon",
            "size_x": 16,
            "size_y": 9,
            "period": 15,
            "difficulty": 3,
            "cells": [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0],
                [1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1],
                [0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
        },
    ]
    NON_OSCILLATORS = [
        {
            "size_x": 3,
            "size_y": 3,
            "cells": [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
        },
        {
            "size_x": 3,
            "size_y": 3,
            "cells": [
                [0, 0, 1],
                [0, 1, 0],
                [1, 0, 0],
            ],
        },
        {
            "size_x": 3,
            "size_y": 3,
            "cells": [
                [1, 0, 0],
                [0, 1, 0],
                [1, 0, 0],
            ],
        },
        {
            "size_x": 3,
            "size_y": 3,
            "cells": [
                [0, 0, 1],
                [0, 1, 0],
                [0, 0, 1],
            ],
        },
        {
            "size_x": 4,
            "size_y": 4,
            "cells": [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
        },
        {
            "size_x": 5,
            "size_y": 5,
            "cells": [
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
            ],
        },
        {
            "size_x": 6,
            "size_y": 6,
            "cells": [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
        },
    ]

    def __getitem__(self, idx: int) -> dict:
        """Generate a single GameOfLife task

        Returns:
            dict with keys:
                - question: str, the task description
                - answer: str, a solution string
                - metadata: dict with generation parameters
        """
        # Create a reproducible random generator for this index.
        rng = Random(self.seed + idx)

        # Flip a coin to decide if we should oscillate.
        should_oscillate = rng.choice([True, False])

        # Get dimensions for convenience.
        grid_x = self.config.grid_size_x
        grid_y = self.config.grid_size_y

        # Initialize the board.
        # Note: cpl.init_simple2d returns an array with shape
        # (timesteps, grid_size_x, grid_size_y)
        board = cpl.init_simple2d(grid_x, grid_y)
        board[:, :, :] = 0  # reset all cells to dead

        # We will place patterns on the initial board (timestep 0).
        initial_board = board[0]

        # Create an occupancy grid to keep track of which cells (and their 1-cell buffer)
        # are already occupied by a pattern.
        occupancy = [[False for _ in range(grid_y)] for _ in range(grid_x)]

        # Determine which set of patterns to use based on should_oscillate.
        if should_oscillate:
            valid_patterns = [osc for osc in self.OSCILLATORS if osc["difficulty"] == self.config.difficulty]
        else:
            valid_patterns = self.NON_OSCILLATORS

        placed_patterns: List[Dict] = []

        # Place the requested number of patterns.
        for _ in range(self.config.num_oscillators):
            pattern = rng.choice(valid_patterns)
            height = pattern["size_y"]
            width = pattern["size_x"]

            # Ensure the pattern (plus a 1-cell border) fits in the grid.
            # Valid top-left positions (i,j) must satisfy:
            # 1 <= i <= grid_x - height - 1 and 1 <= j <= grid_y - width - 1.
            attempts = 1000
            placed = False
            while attempts > 0 and not placed:
                i = rng.randint(1, grid_x - height - 1)
                j = rng.randint(1, grid_y - width - 1)

                # Check if the region from (i-1, j-1) to (i+height, j+width) is free.
                valid = True
                for x in range(i - 1, i + height + 1):
                    for y in range(j - 1, j + width + 1):
                        if occupancy[x][y]:
                            valid = False
                            break
                    if not valid:
                        break

                if valid:
                    # Mark the region (including the 1-cell border) as occupied.
                    for x in range(i - 1, i + height + 1):
                        for y in range(j - 1, j + width + 1):
                            occupancy[x][y] = True

                    # Place the pattern on the initial board.
                    for dx in range(height):
                        for dy in range(width):
                            initial_board[i + dx, j + dy] = pattern["cells"][dx][dy]

                    placed = True
                    placed_patterns.append({"name": pattern.get("name", "non-oscillator"), "position": (i, j)})

                attempts -= 1
            # If no valid placement is found after many attempts, we skip this pattern.

        # Convert the initial board state to string
        board_str = str(initial_board)

        # Create the question string.
        question = (
            f"This is a 'Game of Life' grid. We consider a game halted if there are no cells alive.\n"
            f"Will this game halt at or before {self.config.max_simulation_steps} steps? Assume a Moore neighborhood and wrapping topology. If it will halt, reply 'True'. If it won't halt, reply 'False'.\n\n"
            f"Initial board:\n{board_str}"
        )

        return {
            "question": question,
            "answer": str(not should_oscillate),
            "metadata": {
                "grid_size_x": grid_x,
                "grid_size_y": grid_y,
                "placed_patterns": placed_patterns,
                "simulation_steps": self.config.max_simulation_steps,
                "should_oscillate": should_oscillate,
            },
        }

    def score_answer(self, answer: Optional[str], entry: Dict[str, any]) -> float:
        """Determine if the solution provided solves the GoL task.

        The function awards 1.0 for a correct answer.

        Args:
            answer (Optional[str]): The user's answer.
            entry (Dict[str, any]): The original dataset entry containing the correct answer.

        Returns:
            float: The computed score between 0.0 and 1.0.
        """

        if answer is not None and bool(answer) == bool(entry["answer"]):
            # python's bool conversion is very tolerant and normally doesn't raise exceptions
            return 1.0
        return 0.0


register_dataset("game_of_life_halting", GameOfLifeHaltingDataset, GameOfLifeHaltingConfig)
