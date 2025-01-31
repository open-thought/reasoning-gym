import random
import string
from dataclasses import dataclass
from typing import Optional

from ..factory import ProceduralDataset, register_dataset


@dataclass
class MazeConfig:
    """
    Configuration for the maze task.

    - min_dist: Minimum distance from start to goal (shortest path).
    - max_dist: Maximum distance from start to goal (shortest path).
    - min_grid_size: Minimum dimension of the square grid.
    - max_grid_size: Maximum dimension of the square grid.
    - seed: Optional seed for reproducibility.
    - size: Number of maze samples in the virtual dataset.
    """

    min_dist: int = 5
    max_dist: int = 10
    min_grid_size: int = 5
    max_grid_size: int = 10
    seed: Optional[int] = None
    size: int = 50

    def validate(self) -> None:
        """Validate configuration parameters."""
        assert self.min_dist >= 1, "min_dist must be >= 1"
        assert self.max_dist >= self.min_dist, "max_dist must be >= min_dist"
        assert self.min_grid_size >= 2, "min_grid_size must be >= 2"
        assert self.max_grid_size >= self.min_grid_size, "max_grid_size must be >= min_grid_size"


class MazeDataset(ProceduralDataset):
    """
    Generates mazes with guaranteed shortest path distance from start to goal
    within [min_dist, max_dist].
    """

    def __init__(
        self,
        config: MazeConfig,
        prob_path=0.7,
        num_retries=1000,
    ):
        super().__init__(config=config, seed=config.seed, size=config.size)
        # Probability that a cell is a path instead of a wall
        self.prob_path = prob_path
        # Number of times to resample a grid to find a suitable maze before giving up
        # This is to avoid infinite loops
        self.num_retries = num_retries

    def __getitem__(self, idx: int) -> dict:
        """
        Generate a single maze item with a BFS shortest path
        distance in [min_dist, max_dist].

        Returns:
            dict with:
            - "question": Maze ASCII text prompting for path length
            - "answer": str, the length of the shortest path
            - "metadata": includes the grid, BFS distance, etc.
        """
        rng = random.Random(self.seed + idx)
        # Characters to use in the maze
        self.wall_char, self.path_char, self.start_char, self.goal_char = self._get_random_chars(rng, n=4)

        for _attempt in range(self.num_retries):
            # Choose random grid size and build a random maze
            size = rng.randint(self.config.min_grid_size, self.config.max_grid_size)
            maze_grid = self._generate_random_maze(rng, size)

            # Place Start (S) and Goal (G) off the border
            start_r, start_c = self._random_floor_cell(rng, maze_grid)
            maze_grid[start_r][start_c] = self.start_char

            goal_r, goal_c = self._random_floor_cell(rng, maze_grid)
            # Ensure it's not the same as Start
            while (goal_r, goal_c) == (start_r, start_c):
                goal_r, goal_c = self._random_floor_cell(rng, maze_grid)
            maze_grid[goal_r][goal_c] = self.goal_char

            # Compute BFS shortest path
            dist = self._bfs_shortest_path(maze_grid, start_r, start_c, goal_r, goal_c)

            if dist is not None and self.config.min_dist <= dist <= self.config.max_dist:
                # Maze is good, build the question
                question_str = (
                    f"Navigate from '{self.start_char}' (start) to '{self.goal_char}' (goal):\n\n"
                    + "```\n"
                    + self._maze_to_str(maze_grid)
                    + "\n```"
                    + "\nLegend: "
                    + f"'{self.wall_char}' = Wall, '{self.path_char}' = Passage\n\n"
                    + "What is the minimum number of steps to reach the goal?"
                )

                return {
                    "question": question_str,
                    "answer": str(dist),
                    "metadata": {
                        "grid_size": size,
                        "grid": ["".join(row) for row in maze_grid],
                        "shortest_path_length": dist,
                        "start": self.start_char,
                        "goal": self.goal_char,
                        "wall": self.wall_char,
                        "path": self.path_char,
                    },
                }

        # If we can't find a suitable maze after self.num_retries attempts, raise an error
        raise RuntimeError(
            f"Could not generate a maze with distance in [{self.config.min_dist}, "
            f"{self.config.max_dist}] after {self.num_retries} attempts."
        )

    def _get_random_chars(self, rng: random.Random, n: int = 4) -> list[str]:
        """Get a list of unique random visible ASCII characters."""
        return rng.sample(string.printable[:94], n)

    def _generate_random_maze(self, rng: random.Random, size: int) -> list[list[str]]:
        """
        Create a maze of dimension 'size' x 'size' filled with random paths or walls.
        We'll keep the outer border as walls to constrain the agent inside.
        """
        grid = [[self.wall_char for _ in range(size)] for _ in range(size)]

        for r in range(1, size - 1):
            for c in range(1, size - 1):
                # Randomly decide if cell is wall or path
                if rng.random() < self.prob_path:
                    grid[r][c] = self.path_char
                else:
                    grid[r][c] = self.wall_char

        return grid

    def _random_floor_cell(self, rng: random.Random, grid: list[list[str]]) -> tuple[int, int]:
        """Pick a random path cell inside the maze (not the border)."""
        size = len(grid)
        while True:
            r = rng.randint(1, size - 2)
            c = rng.randint(1, size - 2)
            if grid[r][c] == self.path_char:
                return (r, c)

    def _bfs_shortest_path(
        self, grid: list[list[str]], start_r: int, start_c: int, goal_r: int, goal_c: int
    ) -> Optional[int]:
        """
        Returns the length of the shortest path from (start_r, start_c)
        to (goal_r, goal_c) using BFS. If no path, return None.
        """
        size = len(grid)
        visited = [[False] * size for _ in range(size)]
        queue = [(start_r, start_c, 0)]  # (row, col, distance)

        visited[start_r][start_c] = True

        # Directions: up, right, down, left
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        while queue:
            r, c, dist = queue.pop(0)
            if (r, c) == (goal_r, goal_c):
                return dist  # Found the shortest path

            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size:
                    if not visited[nr][nc] and grid[nr][nc] != self.wall_char:
                        visited[nr][nc] = True
                        queue.append((nr, nc, dist + 1))

        return None  # No path found

    def _maze_to_str(self, grid: list[list[str]]) -> str:
        """Convert grid to a multiline string representation."""
        return "\n".join("".join(row) for row in grid)


register_dataset("maze", MazeDataset, MazeConfig)
