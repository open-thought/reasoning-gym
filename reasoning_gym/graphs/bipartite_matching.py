"""Find a maximum assignment of workers to compatible jobs."""

import json
from collections import deque
from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from ..coaching import BaseCurriculum, RangeAttributeDefinition
from ..factory import ProceduralDataset, register_dataset

DATASET_NAME = "bipartite_matching"

QUESTION_TEMPLATE = """Assign as many workers as possible to compatible jobs.
There are {num_workers} workers labeled 0 through {last_worker} and {num_jobs} jobs labeled 0 through {last_job}.
Worker labels and job labels refer to separate groups.

Each worker can do only the jobs listed below:
{compatibility}

Each worker may receive at most one job, and each job may be assigned to at most one worker.
You may leave workers and jobs unassigned. Maximize the total number of assigned workers.

Return only a JSON list of [worker, job] pairs. Any maximum assignment is accepted, in any order.
Return [] if no assignments are possible.
"""


@dataclass
class BipartiteMatchingConfig:
    """Configuration for maximum bipartite matching problems."""

    min_num_workers: int = 4
    max_num_workers: int = 8
    min_num_jobs: int = 4
    max_num_jobs: int = 8
    edge_probability: float = 0.35
    size: int = 500
    seed: Optional[int] = None

    def validate(self) -> None:
        counts = (self.min_num_workers, self.max_num_workers, self.min_num_jobs, self.max_num_jobs, self.size)
        assert all(type(value) is int for value in counts), "Worker counts, job counts and size must be integers"
        assert 1 <= self.min_num_workers <= self.max_num_workers, "Require 1 <= min_num_workers <= max_num_workers"
        assert 1 <= self.min_num_jobs <= self.max_num_jobs, "Require 1 <= min_num_jobs <= max_num_jobs"
        assert 0 <= self.edge_probability <= 1, "edge_probability must be between 0 and 1"
        assert self.size > 0, "size must be positive"


class BipartiteMatchingDataset(ProceduralDataset):
    """Generate compatibility graphs and verify maximum-cardinality matchings."""

    def __init__(self, config: BipartiteMatchingConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _maximum_matching(self, adjacency: list[list[int]]) -> list[list[int]]:
        """Find augmenting paths with BFS, avoiding recursion on large graphs."""
        worker_to_job: dict[int, int] = {}
        job_to_worker: dict[int, int] = {}

        for start_worker in range(len(adjacency)):
            queue = deque([start_worker])
            visited_workers = {start_worker}
            parent: dict[int, int] = {}
            free_job = None

            while queue and free_job is None:
                worker = queue.popleft()
                for job in adjacency[worker]:
                    if job in parent:
                        continue
                    parent[job] = worker
                    if job not in job_to_worker:
                        free_job = job
                        break
                    matched_worker = job_to_worker[job]
                    if matched_worker not in visited_workers:
                        visited_workers.add(matched_worker)
                        queue.append(matched_worker)

            # Reverse the alternating path to add one match without losing any.
            while free_job is not None:
                worker = parent[free_job]
                previous_job = worker_to_job.get(worker)
                worker_to_job[worker] = free_job
                job_to_worker[free_job] = worker
                free_job = previous_job

        return [[worker, job] for worker, job in sorted(worker_to_job.items())]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rng = Random(self.seed + idx)
        num_workers = rng.randint(self.config.min_num_workers, self.config.max_num_workers)
        num_jobs = rng.randint(self.config.min_num_jobs, self.config.max_num_jobs)
        adjacency = [
            [job for job in range(num_jobs) if rng.random() < self.config.edge_probability] for _ in range(num_workers)
        ]
        solution = self._maximum_matching(adjacency)
        compatibility = "\n".join(
            f"Worker {worker}: {', '.join(map(str, jobs)) if jobs else 'none'}" for worker, jobs in enumerate(adjacency)
        )

        return {
            "question": QUESTION_TEMPLATE.format(
                num_workers=num_workers,
                last_worker=num_workers - 1,
                num_jobs=num_jobs,
                last_job=num_jobs - 1,
                compatibility=compatibility,
            ),
            "answer": json.dumps(solution),
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "num_workers": num_workers,
                "num_jobs": num_jobs,
                "adjacency": adjacency,
                "solution": solution,
                "matching_size": len(solution),
                "difficulty": {
                    "num_workers": (self.config.min_num_workers, self.config.max_num_workers),
                    "num_jobs": (self.config.min_num_jobs, self.config.max_num_jobs),
                    "edge_probability": self.config.edge_probability,
                },
            },
        }

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """Accept any valid maximum matching, rather than one reference ordering."""
        if not isinstance(answer, str):
            return 0.0
        try:
            pairs = json.loads(answer)
        except (ValueError, RecursionError):
            return 0.0

        metadata = entry["metadata"]
        if not isinstance(pairs, list) or len(pairs) != metadata["matching_size"]:
            return 0.0

        used_workers, used_jobs = set(), set()
        for pair in pairs:
            if not isinstance(pair, list) or len(pair) != 2 or any(type(label) is not int for label in pair):
                return 0.0
            worker, job = pair
            if not (0 <= worker < metadata["num_workers"] and 0 <= job < metadata["num_jobs"]):
                return 0.0
            if worker in used_workers or job in used_jobs or job not in metadata["adjacency"][worker]:
                return 0.0
            used_workers.add(worker)
            used_jobs.add(job)

        return 1.0


class BipartiteMatchingCurriculum(BaseCurriculum):
    """Increase the sizes of the two groups independently."""

    def __init__(self):
        super().__init__(BipartiteMatchingCurriculum.__name__, BipartiteMatchingConfig)
        self._define_attributes(
            RangeAttributeDefinition(
                name="num_workers",
                levels=[4, 8, 12, 20, 30],
                description="Number of workers",
                lower_field_name="min_num_workers",
                upper_field_name="max_num_workers",
            ),
            RangeAttributeDefinition(
                name="num_jobs",
                levels=[4, 8, 12, 20, 30],
                description="Number of jobs",
                lower_field_name="min_num_jobs",
                upper_field_name="max_num_jobs",
            ),
        )


register_dataset(DATASET_NAME, BipartiteMatchingDataset, BipartiteMatchingConfig, BipartiteMatchingCurriculum)
