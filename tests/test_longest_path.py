"""Tests for the Longest Path grid task."""

from itertools import permutations, product

import pytest

from reasoning_gym import create_dataset, get_score_answer_fn
from reasoning_gym.graphs.longest_path import LongestPathConfig, LongestPathDataset


@pytest.mark.parametrize(
    "overrides",
    [
        {"min_rows": 1},
        {"min_rows": 5, "max_rows": 4},
        {"min_cols": 1},
        {"min_cols": 5, "max_cols": 4},
        {"max_rows": 11},
        {"max_cols": 11},
        {"p_blocked": -0.1},
        {"p_blocked": 1.1},
        {"max_open_cells": 19},
        {"max_open_cells": 8},
        {"size": 0},
    ],
)
def test_config_validation(overrides):
    with pytest.raises(AssertionError):
        LongestPathConfig(**overrides).validate()


@pytest.mark.parametrize(
    "rows, expected",
    [
        (["*O", "O#"], 2),
        (["*OO", "OOO", "OO#"], 8),
        (["*#O", "OOO", "OOO"], 7),
        (["X*X", "OO#"], 2),  # The dead end cannot be visited and retraced.
        (["*OO#", "XXXX"], 3),
        (["*#", "XX"], 1),
        (["*X#"], -1),
    ],
)
def test_solver_known_grids(rows, expected):
    matrix = [list(row) for row in rows]
    assert LongestPathDataset._get_longest_path_length(matrix) == expected


def test_solver_against_exhaustive_paths():
    # Enumerate ordered subsets directly, independently of the memoized solver.
    cells = list(product(range(2), range(3)))
    for start, goal in permutations(cells, 2):
        other_cells = [cell for cell in cells if cell not in (start, goal)]
        for blocked in product((False, True), repeat=len(other_cells)):
            matrix = [["O"] * 3 for _ in range(2)]
            available = []
            for cell, is_blocked in zip(other_cells, blocked):
                r, c = cell
                if is_blocked:
                    matrix[r][c] = "X"
                else:
                    available.append(cell)
            matrix[start[0]][start[1]] = "*"
            matrix[goal[0]][goal[1]] = "#"

            expected = -1
            for length in range(len(available) + 1):
                for intermediate in permutations(available, length):
                    path = (start, *intermediate, goal)
                    if all(abs(r - nr) + abs(c - nc) == 1 for (r, c), (nr, nc) in zip(path, path[1:])):
                        expected = length + 1
            assert LongestPathDataset._get_longest_path_length(matrix) == expected


def test_generation_is_deterministic_and_well_formed():
    config = LongestPathConfig(seed=42, size=30)
    first = LongestPathDataset(config)
    second = LongestPathDataset(config)

    for idx in range(config.size):
        entry = first[idx]
        assert entry == second[idx]
        assert set(entry) == {"question", "answer", "metadata"}
        assert entry["metadata"]["source_dataset"] == "longest_path"

        matrix = entry["metadata"]["matrix"]
        assert config.min_rows <= len(matrix) <= config.max_rows
        cols = len(matrix[0])
        assert config.min_cols <= cols <= config.max_cols
        assert all(len(row) == cols for row in matrix)
        assert {cell for row in matrix for cell in row} <= {"*", "#", "O", "X"}
        assert sum(cell == "*" for row in matrix for cell in row) == 1
        assert sum(cell == "#" for row in matrix for cell in row) == 1
        open_cells = sum(cell != "X" for row in matrix for cell in row)
        assert entry["metadata"]["source_index"] == idx
        assert entry["metadata"]["rows"] == len(matrix)
        assert entry["metadata"]["cols"] == cols
        assert entry["metadata"]["open_cell_count"] == open_cells <= config.max_open_cells
        assert int(entry["answer"]) == entry["metadata"]["longest_path_length"]
        assert 1 <= int(entry["answer"]) < open_cells
        assert "\n".join(" ".join(row) for row in matrix) in entry["question"]
        assert first.score_answer(entry["answer"], entry) == 1.0


@pytest.mark.parametrize("p_blocked", [0.0, 1.0])
@pytest.mark.parametrize("rows, cols, max_open_cells", [(2, 2, 3), (3, 6, 18), (10, 9, 18)])
def test_generation_at_configuration_limits(p_blocked, rows, cols, max_open_cells):
    config = LongestPathConfig(
        min_rows=rows,
        max_rows=rows,
        min_cols=cols,
        max_cols=cols,
        p_blocked=p_blocked,
        max_open_cells=max_open_cells,
        seed=42,
        size=20,
    )
    dataset = LongestPathDataset(config)
    for entry in dataset:
        matrix = entry["metadata"]["matrix"]
        start = next((r, c) for r, row in enumerate(matrix) for c, cell in enumerate(row) if cell == "*")
        goal = next((r, c) for r, row in enumerate(matrix) for c, cell in enumerate(row) if cell == "#")
        # Check reachability independently, including the all-blocked case before carving.
        reachable = {start}
        pending = [start]
        while pending:
            r, c = pending.pop()
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < rows and 0 <= nc < cols and matrix[nr][nc] != "X" and (nr, nc) not in reachable:
                    reachable.add((nr, nc))
                    pending.append((nr, nc))
        assert goal in reachable
        assert entry["metadata"]["open_cell_count"] <= config.max_open_cells
        assert int(entry["answer"]) >= abs(start[0] - goal[0]) + abs(start[1] - goal[1])
        if p_blocked == 1.0:
            assert int(entry["answer"]) == entry["metadata"]["open_cell_count"] - 1


@pytest.mark.parametrize(
    "answer, expected",
    [
        ("8", 1.0),
        (" 8\n", 1.0),
        (None, 0.0),
        (8, 0.0),
        ("", 0.0),
        ("The answer is 8", 0.0),
        ("80", 0.0),
        ("18", 0.0),
        ("7", 0.0),
        ("8.0", 0.0),
        ("-8", 0.0),
    ],
)
def test_score_answer_is_strict(answer, expected):
    dataset = LongestPathDataset(LongestPathConfig(seed=7, size=3))
    assert dataset.score_answer(answer, {"answer": "8"}) == expected


def test_iteration_respects_size():
    config = LongestPathConfig(seed=17, size=5)
    dataset = LongestPathDataset(config)
    assert len(list(dataset)) == 5
    assert list(dataset) == list(dataset)


def test_factory_registration():
    dataset = create_dataset("longest_path", seed=42, size=2)
    assert isinstance(dataset, LongestPathDataset)
    assert dataset.category == "graphs"
    entry = dataset[0]
    assert get_score_answer_fn("longest_path")(entry["answer"], entry) == 1.0
