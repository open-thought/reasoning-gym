import pytest

from reasoning_gym import create_dataset
from reasoning_gym.games.maze import MazeConfig, MazeDataset


def test_maze_config_validation():
    with pytest.raises(AssertionError):
        cfg = MazeConfig(min_dist=0)
        cfg.validate()

    with pytest.raises(AssertionError):
        cfg = MazeConfig(min_dist=10, max_dist=5)
        cfg.validate()

    with pytest.raises(AssertionError):
        cfg = MazeConfig(min_grid_size=1, max_grid_size=5)
        cfg.validate()

    with pytest.raises(AssertionError):
        cfg = MazeConfig(min_grid_size=10, max_grid_size=5)
        cfg.validate()


def test_maze_dataset_creation():
    dataset_size = 22
    cfg = MazeConfig(
        min_dist=3,
        max_dist=5,
        min_grid_size=5,
        max_grid_size=5,
        seed=42,
        size=dataset_size,
    )

    ds = MazeDataset(cfg)

    assert len(ds) == dataset_size


def test_maze_dataset_items():
    ds = create_dataset(
        "maze",
        min_dist=3,
        max_dist=5,
        min_grid_size=5,
        max_grid_size=5,
        size=2,
        seed=42,
    )

    for item in ds:
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item
        meta = item["metadata"]
        assert isinstance(meta["grid"], list)
        assert isinstance(meta["shortest_path_length"], int)


def test_maze_shortest_path_correctness():
    """
    Test that the BFS distance reported by each item is indeed correct and within the limits.
    """
    min_dist = 4
    max_dist = 8
    ds = create_dataset(
        "maze",
        min_dist=min_dist,
        max_dist=max_dist,
        min_grid_size=5,
        max_grid_size=6,
        size=3,
        seed=99,
    )

    for item in ds:
        reported_dist = int(item["answer"])
        grid = item["metadata"]["grid"]  # list of strings

        # Convert grid to 2D array
        maze = [list(row) for row in grid]
        size = len(maze)

        start = None
        goal = None
        for r in range(size):
            for c in range(len(maze[r])):
                if maze[r][c] == item["metadata"]["start"]:
                    start = (r, c)
                elif maze[r][c] == item["metadata"]["goal"]:
                    goal = (r, c)

        assert start is not None, "Start not found!"
        assert goal is not None, "Goal not found!"

        # Now BFS to confirm the distance
        bfs_dist = _bfs_distance(maze, start, goal, item["metadata"]["wall"])
        assert (
            bfs_dist == reported_dist
        ), f"Mismatch in BFS distance: BFS found {bfs_dist}, item reported {reported_dist}.\n" f"Maze:\n" + "\n".join(
            grid
        )
        assert bfs_dist <= max_dist and bfs_dist >= min_dist


def _bfs_distance(maze, start, goal, wall_char):
    """Utility BFS to confirm shortest path length in the test."""
    from collections import deque

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    visited = set([start])
    queue = deque([(start[0], start[1], 0)])  # (row, col, distance)

    while queue:
        r, c, dist = queue.popleft()
        if (r, c) == goal:
            return dist

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < len(maze) and 0 <= nc < len(maze[0]):
                if maze[nr][nc] != wall_char and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc, dist + 1))

    return None  # no path found
