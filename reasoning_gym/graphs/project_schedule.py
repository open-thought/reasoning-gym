"""Compute earliest task completion times in a project with parallel execution."""

import json
from collections import deque
from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from ..coaching import BaseCurriculum, RangeAttributeDefinition, ScalarAttributeDefinition
from ..factory import ProceduralDataset, register_dataset

DATASET_NAME = "project_schedule"

QUESTION_TEMPLATE = """A project has {num_tasks} tasks, labeled 0 through {last_task}.
Time starts at 0. There are unlimited workers, so independent tasks can run in parallel.
Each task runs for its full duration and can start only after ALL its prerequisites have finished.
A task may start at the exact instant its last prerequisite finishes. A task with no prerequisites
can start at time 0. For example, a task of duration 3 that starts at time 0 finishes at time 3.

The prerequisite relationships contain no cycles.

Task | Duration | Prerequisites (task IDs)
{tasks}

Find the earliest finish time of EVERY task.
Return only a JSON array of {num_tasks} integers in task-ID order: task 0, task 1, ..., task {last_task}.
"""


@dataclass
class ProjectScheduleConfig:
    """Configure task count, durations, dependency depth, and maximum fan-in.

    Dependency depth counts tasks on the longest prerequisite chain, including
    both endpoints. It is exact, so it cannot exceed the minimum task count.
    The maximum prerequisite count is a cap; root tasks always have none.
    """

    min_num_tasks: int = 6
    max_num_tasks: int = 12
    min_duration: int = 1
    max_duration: int = 9
    dependency_depth: int = 3
    max_prerequisites: int = 3
    seed: Optional[int] = None
    size: int = 500

    def validate(self) -> None:
        for field in (
            "min_num_tasks",
            "max_num_tasks",
            "min_duration",
            "max_duration",
            "dependency_depth",
            "max_prerequisites",
            "size",
        ):
            assert type(getattr(self, field)) is int, f"{field} must be an integer"
        assert 1 <= self.min_num_tasks <= self.max_num_tasks, "Invalid task count range"
        assert 1 <= self.min_duration <= self.max_duration, "Durations must be positive and ordered"
        assert 1 <= self.dependency_depth <= self.min_num_tasks, "Depth must be between 1 and min_num_tasks"
        assert self.max_prerequisites >= 1, "max_prerequisites must be positive"
        assert self.size > 0, "size must be positive"


class ProjectScheduleDataset(ProceduralDataset):
    """Find every task's earliest finish time given durations and prerequisites.

    Independent tasks execute in parallel with unlimited workers. Instances are
    acyclic and have a unique completion-time vector, computed by topological
    dynamic programming. Shuffled task IDs hide the generating order.
    """

    def __init__(self, config: ProjectScheduleConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _generate_prerequisites(self, rng: Random, num_tasks: int) -> list[list[int]]:
        """Build nonempty layers so the longest chain has exactly the requested depth."""
        task_ids = list(range(num_tasks))
        rng.shuffle(task_ids)

        depth = self.config.dependency_depth
        layer_sizes = [1] * depth
        remaining = num_tasks - depth
        # Two roots make parallel execution and a merge possible when there is room.
        if depth > 1 and remaining > 0:
            layer_sizes[0] += 1
            remaining -= 1
        for _ in range(remaining):
            layer_sizes[rng.randrange(depth)] += 1

        prerequisites = [[] for _ in range(num_tasks)]
        earlier_tasks = []
        previous_layer = []
        offset = 0
        for layer_index, layer_size in enumerate(layer_sizes):
            layer = task_ids[offset : offset + layer_size]
            offset += layer_size
            if layer_index > 0:
                for task in layer:
                    # A predecessor in the previous layer guarantees the exact depth.
                    required = rng.choice(previous_layer)
                    count = rng.randint(1, min(self.config.max_prerequisites, len(earlier_tasks)))
                    # Ensure a join in every instance where the configuration permits one.
                    if layer_index == 1 and task == layer[0] and len(previous_layer) > 1:
                        count = max(count, min(2, self.config.max_prerequisites))
                    candidates = [other for other in earlier_tasks if other != required]
                    prerequisites[task] = sorted([required] + rng.sample(candidates, count - 1))
            earlier_tasks.extend(layer)
            previous_layer = layer
        return prerequisites

    @staticmethod
    def _earliest_finish_times(durations: list[int], prerequisites: list[list[int]]) -> list[int]:
        """Compute completion times using Kahn's traversal in O(V + E) time."""
        successors = [[] for _ in durations]
        indegrees = [len(parents) for parents in prerequisites]
        for task, parents in enumerate(prerequisites):
            for parent in parents:
                successors[parent].append(task)

        ready = deque(task for task, degree in enumerate(indegrees) if degree == 0)
        starts = [0] * len(durations)
        finishes = [0] * len(durations)
        processed = 0
        while ready:
            task = ready.popleft()
            finishes[task] = starts[task] + durations[task]
            processed += 1
            for successor in successors[task]:
                starts[successor] = max(starts[successor], finishes[task])
                indegrees[successor] -= 1
                if indegrees[successor] == 0:
                    ready.append(successor)

        if processed != len(durations):
            raise ValueError("Project prerequisites contain a cycle")
        return finishes

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rng = Random(self.seed + idx)
        num_tasks = rng.randint(self.config.min_num_tasks, self.config.max_num_tasks)
        durations = [rng.randint(self.config.min_duration, self.config.max_duration) for _ in range(num_tasks)]
        prerequisites = self._generate_prerequisites(rng, num_tasks)
        solution = self._earliest_finish_times(durations, prerequisites)
        task_rows = "\n".join(f"{task} | {durations[task]} | {prerequisites[task]}" for task in range(num_tasks))

        return {
            "question": QUESTION_TEMPLATE.format(num_tasks=num_tasks, last_task=num_tasks - 1, tasks=task_rows),
            "answer": json.dumps(solution),
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "durations": durations,
                "prerequisites": prerequisites,
                "solution": solution,
                "difficulty": {
                    "num_tasks": num_tasks,
                    "dependency_depth": self.config.dependency_depth,
                    "max_prerequisites": self.config.max_prerequisites,
                    "duration_range": (self.config.min_duration, self.config.max_duration),
                },
            },
        }

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """Award full credit only for the complete, correct integer completion-time vector."""
        if not isinstance(answer, str):
            return 0.0
        try:
            values = json.loads(answer)
        except (ValueError, RecursionError):
            return 0.0
        if not isinstance(values, list) or any(type(value) is not int for value in values):
            return 0.0
        return float(values == entry["metadata"]["solution"])


class ProjectScheduleCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(ProjectScheduleCurriculum.__name__, ProjectScheduleConfig)
        self._define_attributes(
            RangeAttributeDefinition(
                name="num_tasks",
                levels=[6, 12, 24, 48],
                lower_field_name="min_num_tasks",
                upper_field_name="max_num_tasks",
                ensure_interval=True,
                description="Number of tasks in the project",
            ),
            RangeAttributeDefinition(
                name="duration",
                levels=[1, 9, 20, 50],
                lower_field_name="min_duration",
                upper_field_name="max_duration",
                ensure_interval=True,
                description="Task duration range",
            ),
            ScalarAttributeDefinition(
                name="dependency_depth",
                levels=[2, 3, 4, 6],
                default_level=1,
                field_name="dependency_depth",
                description="Number of tasks on the longest prerequisite chain",
            ),
            ScalarAttributeDefinition(
                name="max_prerequisites",
                levels=[2, 3, 4, 6],
                default_level=1,
                field_name="max_prerequisites",
                description="Maximum prerequisites per task",
            ),
        )


register_dataset(DATASET_NAME, ProjectScheduleDataset, ProjectScheduleConfig, ProjectScheduleCurriculum)
