"""Inspired by https://arxiv.org/pdf/2403.06963"""

import random
from dataclasses import dataclass
from typing import Any, Optional

from ..coaching import BaseCurriculum, RangeAttributeDefinition, ScalarAttributeDefinition
from ..factory import ProceduralDataset, register_dataset

DATASET_NAME = "path_star"


@dataclass
class PathStarConfig:
    degree: int = 3
    node_range: int = 100_000
    min_path_length: int = 3
    max_path_length: int = 5

    reversed: bool = False
    teacherless: bool = False

    size: int = 500  # Virtual dataset size
    seed: Optional[int] = None

    def validate(self) -> None:
        assert self.degree >= 2 and self.path_length >= 1
        assert self.node_range > self.degree * self.path_length + 1


class PathStarDataset(ProceduralDataset):
    """Procedurally generates path-star graph problems."""

    def __init__(self, config: PathStarConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rng = random.Random(self.seed + idx)

        cfg: PathStarConfig = self.config
        center = rng.randrange(cfg.node_range)
        path_length = rng.randint(cfg.min_path_length, cfg.max_path_length)

        # allocate unique node labels
        paths = []
        used = {center}
        for _ in range(cfg.degree):
            path = []
            for _ in range(path_length):
                n = rng.randrange(cfg.node_range)
                while n in used:
                    n = rng.randrange(cfg.node_range)
                used.add(n)
                path.append(n)
            paths.append(path)

        goal_path = rng.choice(paths)
        goal = goal_path[-1]

        # build edge list
        edges = [(center, p[0]) for p in paths]
        for p in paths:
            edges.extend(zip(p[:-1], p[1:]))
        rng.shuffle(edges)

        # serialise graph
        lines = [f"e {u} {v}" for u, v in edges]
        lines.append(f"S {center}")
        lines.append(f"G {goal}")
        lines.append("?")
        prefix = "\n".join(lines)

        # gold path
        gold = [center] + goal_path
        if cfg.reversed:
            gold = list(reversed(gold))
        answer = " ".join(map(str, gold))

        if cfg.teacherless:
            q = "X" * len(prefix)  # same length dummy token(s)
        else:
            q = prefix

        return {
            "question": q,
            "answer": answer,
            "metadata": {
                "center": center,
                "goal": goal,
                "path_length": path_length,
                "goal_path": gold if not cfg.reversed else list(reversed(gold)),
                "difficulty": {
                    "degree": cfg.degree,
                    "node_range": cfg.node_range,
                    "path_length": (cfg.min_path_length, cfg.max_path_length),
                },
            },
        }


class PathStarCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(PathStarCurriculum.__name__, PathStarConfig)

        # Define attributes
        self._define_attributes(
            ScalarAttributeDefinition(
                name="degree",
                levels=[2, 3, 4, 5],
                description="Degree of the graph",
                field_name="degree",
            ),
            ScalarAttributeDefinition(
                name="node_range",
                levels=[10_000, 50_000, 100_000, 200_000],
                description="Range of node labels",
                field_name="node_range",
            ),
            RangeAttributeDefinition(
                name="path_length",
                levels=[3, 5, 6, 7],
                description="Length of paths in the graph",
                lower_field_name="min_path_length",
                upper_field_name="max_path_length",
            ),
        )


register_dataset(DATASET_NAME.PathStarDataset, PathStarConfig, PathStarCurriculum)
