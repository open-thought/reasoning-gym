import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Optional

from ..coaching import BaseCurriculum, ScalarAttributeDefinition
from ..factory import ProceduralDataset, register_dataset

DATASET_NAME = "job_scheduling"

TASK_TYPES = ("critical_path", "interval_scheduling", "task_ordering")


@dataclass
class JobSchedulingConfig:
    min_jobs: int = 4
    max_jobs: int = 7
    min_duration: int = 1
    max_duration: int = 10
    task_types: tuple[str, ...] = TASK_TYPES
    task_weights: list[float] = field(default_factory=lambda: [0.34, 0.33, 0.33])
    seed: Optional[int] = None
    size: int = 500

    def validate(self) -> None:
        assert self.size > 0, "size must be positive"
        assert self.min_jobs >= 3, "min_jobs must be >= 3"
        assert self.max_jobs >= self.min_jobs, "max_jobs must be >= min_jobs"
        assert self.min_duration >= 1, "min_duration must be >= 1"
        assert self.max_duration >= self.min_duration, "max_duration must be >= min_duration"
        assert len(self.task_types) > 0, "must have at least one task type"
        assert all(t in TASK_TYPES for t in self.task_types), f"invalid task type"
        assert len(self.task_weights) == len(self.task_types), "weights must match types"


def _critical_path(jobs: dict, deps: dict) -> int:
    earliest = {}
    in_deg = defaultdict(int)
    for j in jobs:
        in_deg[j] = 0
    for j, prereqs in deps.items():
        in_deg[j] = len(prereqs)

    queue = deque()
    for j in jobs:
        if in_deg[j] == 0:
            earliest[j] = 0
            queue.append(j)

    adj = defaultdict(list)
    for j, prereqs in deps.items():
        for p in prereqs:
            adj[p].append(j)

    while queue:
        j = queue.popleft()
        for nxt in adj[j]:
            start = earliest[j] + jobs[j]
            earliest[nxt] = max(earliest.get(nxt, 0), start)
            in_deg[nxt] -= 1
            if in_deg[nxt] == 0:
                queue.append(nxt)

    return max(earliest[j] + jobs[j] for j in jobs)


def _topo_sort(jobs: list, deps: dict) -> list:
    in_deg = {j: 0 for j in jobs}
    adj = defaultdict(list)
    for j, prereqs in deps.items():
        in_deg[j] = len(prereqs)
        for p in prereqs:
            adj[p].append(j)

    queue = deque(sorted(j for j in jobs if in_deg[j] == 0))
    result = []
    while queue:
        j = queue.popleft()
        result.append(j)
        for nxt in sorted(adj[j]):
            in_deg[nxt] -= 1
            if in_deg[nxt] == 0:
                queue.append(nxt)
    return result


class JobSchedulingDataset(ProceduralDataset):
    def __init__(self, config: JobSchedulingConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _make_critical_path(self, rng: random.Random) -> dict:
        n = rng.randint(self.config.min_jobs, self.config.max_jobs)
        names = [chr(65 + i) for i in range(n)]
        jobs = {name: rng.randint(self.config.min_duration, self.config.max_duration) for name in names}
        deps = {}
        for i in range(1, n):
            num_deps = rng.randint(0, min(2, i))
            if num_deps > 0:
                deps[names[i]] = rng.sample(names[:i], num_deps)

        cp = _critical_path(jobs, deps)
        job_desc = ", ".join(f"{name}(duration={d})" for name, d in jobs.items())
        dep_desc = "; ".join(f"{j} depends on {', '.join(p)}" for j, p in deps.items()) or "no dependencies"
        question = (
            f"Given jobs: {job_desc}. Dependencies: {dep_desc}. "
            f"All jobs without dependencies can start immediately and run in parallel. "
            f"What is the minimum total time to complete all jobs? "
            f"Give your answer as a single integer."
        )
        return {"question": question, "answer": str(cp), "task_type": "critical_path"}

    def _make_interval_scheduling(self, rng: random.Random) -> dict:
        n = rng.randint(self.config.min_jobs, self.config.max_jobs + 3)
        intervals = []
        for _ in range(n):
            start = rng.randint(0, 20)
            end = start + rng.randint(1, 8)
            intervals.append((start, end))
        intervals.sort(key=lambda x: x[1])

        selected = []
        last_end = -1
        for s, e in intervals:
            if s >= last_end:
                selected.append((s, e))
                last_end = e

        rng.shuffle(intervals)
        intervals_str = ", ".join(f"({s}, {e})" for s, e in intervals)
        question = (
            f"Given the following intervals (start, end): [{intervals_str}]. "
            f"What is the maximum number of non-overlapping intervals you can select? "
            f"Give your answer as a single integer."
        )
        return {"question": question, "answer": str(len(selected)), "task_type": "interval_scheduling"}

    def _make_task_ordering(self, rng: random.Random) -> dict:
        n = rng.randint(self.config.min_jobs, self.config.max_jobs)
        names = [chr(65 + i) for i in range(n)]
        deps = {}
        for i in range(1, n):
            num_deps = rng.randint(0, min(2, i))
            if num_deps > 0:
                deps[names[i]] = rng.sample(names[:i], num_deps)

        order = _topo_sort(names, deps)
        dep_desc = "; ".join(f"{j} depends on {', '.join(p)}" for j, p in deps.items()) or "no dependencies"
        answer = ", ".join(order)
        question = (
            f"Given tasks: {', '.join(names)}. Dependencies: {dep_desc}. "
            f"Give a valid execution order that respects all dependencies. "
            f"List the tasks separated by commas."
        )
        return {"question": question, "answer": answer, "task_type": "task_ordering", "deps": deps, "names": names}

    def __getitem__(self, idx: int) -> dict:
        rng = random.Random(self.seed + idx)
        task_type = rng.choices(self.config.task_types, weights=self.config.task_weights, k=1)[0]

        generators = {
            "critical_path": self._make_critical_path,
            "interval_scheduling": self._make_interval_scheduling,
            "task_ordering": self._make_task_ordering,
        }
        result = generators[task_type](rng)
        return {
            "question": result["question"],
            "answer": result["answer"],
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "task_type": result["task_type"],
                "difficulty": {
                    "min_jobs": self.config.min_jobs,
                    "max_jobs": self.config.max_jobs,
                },
                **({"deps": result["deps"], "names": result["names"]} if "deps" in result else {}),
            },
        }

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        if answer is None:
            return 0.0
        oracle = entry["answer"]
        if answer.strip() == oracle.strip():
            return 1.0
        task_type = entry["metadata"]["task_type"]

        if task_type == "task_ordering":
            try:
                order = [x.strip() for x in answer.split(",")]
                deps = entry["metadata"]["deps"]
                names = entry["metadata"]["names"]
                if set(order) != set(names):
                    return 0.0
                pos = {name: i for i, name in enumerate(order)}
                for j, prereqs in deps.items():
                    for p in prereqs:
                        if pos.get(p, float("inf")) >= pos.get(j, -1):
                            return 0.0
                return 1.0
            except (ValueError, TypeError):
                return 0.0

        try:
            return 1.0 if int(answer.strip()) == int(oracle.strip()) else 0.0
        except ValueError:
            return 0.0


class JobSchedulingCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(JobSchedulingCurriculum.__name__, JobSchedulingConfig)
        self._define_attributes(
            ScalarAttributeDefinition(
                name="max_jobs",
                field_name="max_jobs",
                levels=[4, 7, 10, 15],
                description="Maximum number of jobs",
            ),
        )


register_dataset(DATASET_NAME, JobSchedulingDataset, JobSchedulingConfig, JobSchedulingCurriculum)
