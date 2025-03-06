import random
from typing import Any

from .dsl import *


def strip_prefix(string: str, prefix: str) -> str:
    """
    removes prefix
    """
    return string[len(prefix) :]


def get_generators(generators) -> dict:
    """
    returns mapper from task identifiers (keys) to example generator functions
    """
    prefix = "generate_"
    return {strip_prefix(n, prefix): getattr(generators, n) for n in dir(generators) if n.startswith(prefix)}


def get_verifiers(verifiers) -> dict:
    """
    returns mapper from task identifiers (keys) to example verifier functions
    """
    prefix = "verify_"
    return {strip_prefix(n, prefix): getattr(verifiers, n) for n in dir(verifiers) if n.startswith(prefix)}


def get_pso_difficulty(example: dict) -> float:
    """
    PSO-Difficulty: proxy measure for example difficulty, defined as weighted sum of #Pixels, #Symbols, #Objects
    """
    i, o = example["input"], example["output"]
    hwi = height(i) * width(i)
    hwo = height(o) * width(o)
    pix_pct = (hwi + hwo) / 1800
    col_pct = len(palette(i) | palette(o)) / 10
    obj_dens = (len(objects(i, T, F, F)) / hwi + len(objects(o, T, F, F)) / hwo) / 2
    return (pix_pct + col_pct + obj_dens) / 3


def unifint(rng: random.Random, diff_lb: float, diff_ub: float, bounds: tuple[int, int]) -> int:
    """
    rng
    diff_lb: lower bound for difficulty, must be in range [0, diff_ub]
    diff_ub: upper bound for difficulty, must be in range [diff_lb, 1]
    bounds: interval [a, b] determining the integer values that can be sampled
    """
    a, b = bounds
    d = rng.uniform(diff_lb, diff_ub)
    if not hasattr(rng, "difficulty_samples"):
        rng.difficulty_samples = []
    rng.difficulty_samples.append(d)
    return min(max(a, round(a + (b - a) * d)), b)


def is_grid(grid: Any) -> bool:
    """
    returns True if and only if argument is a valid grid
    """
    if not isinstance(grid, tuple):
        return False
    if not 0 < len(grid) <= 30:
        return False
    if not all(isinstance(r, tuple) for r in grid):
        return False
    if not all(0 < len(r) <= 30 for r in grid):
        return False
    if not len(set(len(r) for r in grid)) == 1:
        return False
    if not all(all(isinstance(x, int) for x in r) for r in grid):
        return False
    if not all(all(0 <= x <= 9 for x in r) for r in grid):
        return False
    return True


def strip_prefix(string: str, prefix: str) -> str:
    """
    removes prefix
    """
    return string[len(prefix) :]


def format_grid(grid: list[list[int]]) -> Grid:
    """
    grid type casting
    """
    return tuple(tuple(row) for row in grid)


def format_example(example: dict) -> dict:
    """
    example data type
    """
    return {"input": format_grid(example["input"]), "output": format_grid(example["output"])}


def format_task(task: dict) -> dict:
    """
    task data type
    """
    return {
        "train": [format_example(example) for example in task["train"]],
        "test": [format_example(example) for example in task["test"]],
    }


def fix_bugs(dataset: dict) -> None:
    """
    fixes bugs in the original ARC training dataset
    """
    dataset["a8d7556c"]["train"][2]["output"] = fill(dataset["a8d7556c"]["train"][2]["output"], 2, {(8, 12), (9, 12)})
    dataset["6cf79266"]["train"][2]["output"] = fill(
        dataset["6cf79266"]["train"][2]["output"], 1, {(6, 17), (7, 17), (8, 15), (8, 16), (8, 17)}
    )
    dataset["469497ad"]["train"][1]["output"] = fill(
        dataset["469497ad"]["train"][1]["output"], 7, {(5, 12), (5, 13), (5, 14)}
    )
    dataset["9edfc990"]["train"][1]["output"] = fill(dataset["9edfc990"]["train"][1]["output"], 1, {(6, 13)})
    dataset["e5062a87"]["train"][1]["output"] = fill(
        dataset["e5062a87"]["train"][1]["output"], 2, {(1, 3), (1, 4), (1, 5), (1, 6)}
    )
    dataset["e5062a87"]["train"][0]["output"] = fill(
        dataset["e5062a87"]["train"][0]["output"], 2, {(5, 2), (6, 3), (3, 6), (4, 7)}
    )
