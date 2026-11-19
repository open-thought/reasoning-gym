"""Weighted job scheduling with tight deadlines (single machine, maximise on-time weight).

Each job has a processing time, a deadline and a weight. The machine runs one job at a time, starting at
time 0, without interruption. A chosen job must finish by its deadline. The goal is to choose and order
jobs so that the total weight of the chosen jobs is as large as possible (scheduling notation: 1 || sum w_j U_j).

Deadlines are generated with the tardiness-factor / due-date-range method, so that only part of the jobs fit.
The optimum is found with the Lawler-Moore dynamic program. Several schedules can reach the optimum, so the
verifier simulates the proposed schedule instead of comparing strings.
"""

import re
from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from ..coaching import BaseCurriculum, RangeAttributeDefinition, ScalarAttributeDefinition
from ..factory import ProceduralDataset, register_dataset

DATASET_NAME = "weighted_job_scheduling"

QUESTION_TEMPLATE = """You have one machine and the following jobs. Each job has a processing time, a deadline and a weight.

{jobs}

Rules:
1. The machine starts at time 0 and runs one job at a time.
2. A job runs from start to finish without interruption, and the next job starts as soon as the previous one ends.
3. Every job you choose must finish at or before its deadline. Jobs you leave out earn nothing.
4. Choose jobs and an order that maximise the total weight of the chosen jobs.

Give the chosen job IDs in the order they run, separated by commas (for example: J3, J1, J5). Do not include any other text."""

_JOB_ID_RE = re.compile(r"J(\d+)")

MAX_GENERATION_ATTEMPTS = 100


@dataclass
class WeightedJobSchedulingConfig:
    """Configuration for weighted job scheduling task generation"""

    min_jobs: int = 4
    max_jobs: int = 8
    min_processing_time: int = 1
    max_processing_time: int = 10
    min_weight: int = 1
    max_weight: int = 20
    tardiness_factor: float = 0.6  # higher -> earlier deadlines -> fewer jobs fit
    due_date_range: float = 0.4  # higher -> deadlines more spread out
    require_greedy_gap: bool = True  # reject instances solved optimally by the weight/time greedy rule
    partial_credit: bool = True  # reward feasible but suboptimal schedules with less than 0.5
    seed: Optional[int] = None
    size: int = 500

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert 2 <= self.min_jobs, "min_jobs must be at least 2"
        assert self.min_jobs <= self.max_jobs, "max_jobs must be >= min_jobs"
        assert 1 <= self.min_processing_time, "min_processing_time must be at least 1"
        assert (
            self.min_processing_time <= self.max_processing_time
        ), "max_processing_time must be >= min_processing_time"
        assert 1 <= self.min_weight, "min_weight must be at least 1"
        assert self.min_weight <= self.max_weight, "max_weight must be >= min_weight"
        assert 0 < self.tardiness_factor < 1, "tardiness_factor must be in (0, 1)"
        assert 0 <= self.due_date_range < 1, "due_date_range must be in [0, 1)"
        assert self.tardiness_factor > self.due_date_range / 2, "tardiness_factor must exceed due_date_range / 2"
        assert self.size > 0, "size must be positive"


def simulate(order: list[dict]) -> tuple[bool, int]:
    """Run jobs back to back from time 0.

    Args:
        order: job dicts (keys id, p, d, w) in the order they run
    Returns:
        (all_on_time, total_weight of the jobs run); stops at the first late job
    """
    t = 0
    total = 0
    for job in order:
        t += job["p"]
        if t > job["d"]:
            return False, total
        total += job["w"]
    return True, total


def solve_optimal(jobs: list[dict]) -> tuple[int, list[dict]]:
    """Exact Lawler-Moore dynamic program.

    A set of jobs can all finish on time iff it can in earliest-deadline-first order, so we sort by deadline and run a
    knapsack over total processing time. Returns (optimal_weight, chosen jobs in deadline order).
    """
    ordered = sorted(jobs, key=lambda j: (j["d"], j["id"]))
    horizon = max(j["d"] for j in ordered)
    neg = -1
    # table[i][t]: best weight using the first i jobs with total processing time exactly t (neg = unreachable)
    table = [[neg] * (horizon + 1) for _ in range(len(ordered) + 1)]
    table[0][0] = 0
    for i, job in enumerate(ordered, start=1):
        prev, cur = table[i - 1], table[i]
        for t in range(horizon + 1):
            cur[t] = prev[t]
            if job["p"] <= t <= job["d"] and prev[t - job["p"]] != neg:
                cand = prev[t - job["p"]] + job["w"]
                if cand > cur[t]:
                    cur[t] = cand

    last = table[len(ordered)]
    best_t = max(range(horizon + 1), key=lambda t: (last[t], -t))
    best = last[best_t]

    chosen = []
    t = best_t
    for i in range(len(ordered), 0, -1):
        if table[i][t] != table[i - 1][t]:
            chosen.append(ordered[i - 1])
            t -= ordered[i - 1]["p"]
    chosen.reverse()
    return best, chosen


def greedy_weight(jobs: list[dict]) -> int:
    """Weight reached by the greedy rule: highest weight per unit time first, keep a job if the set still fits."""
    kept: list[dict] = []
    for job in sorted(jobs, key=lambda j: (-j["w"] / j["p"], j["id"])):
        candidate = sorted(kept + [job], key=lambda j: (j["d"], j["id"]))
        if simulate(candidate)[0]:
            kept.append(job)
    return sum(j["w"] for j in kept)


class WeightedJobSchedulingDataset(ProceduralDataset):
    """Generates weighted job scheduling problems with tight deadlines"""

    def __init__(self, config: WeightedJobSchedulingConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _draw_jobs(self, rng: Random) -> list[dict]:
        cfg = self.config
        n = rng.randint(cfg.min_jobs, cfg.max_jobs)
        procs = [rng.randint(cfg.min_processing_time, cfg.max_processing_time) for _ in range(n)]
        weights = [rng.randint(cfg.min_weight, cfg.max_weight) for _ in range(n)]
        total = sum(procs)
        lo_base = int(total * (1 - cfg.tardiness_factor - cfg.due_date_range / 2))
        hi_base = int(total * (1 - cfg.tardiness_factor + cfg.due_date_range / 2))
        jobs = []
        for i in range(n):
            lo = max(procs[i], lo_base)
            hi = max(lo, hi_base)
            jobs.append({"id": f"J{i + 1}", "p": procs[i], "d": rng.randint(lo, hi), "w": weights[i]})
        return jobs

    def __getitem__(self, idx: int) -> dict:
        rng = Random(self.seed + idx)
        cfg = self.config

        fallback = None
        chosen = None
        for _ in range(MAX_GENERATION_ATTEMPTS):
            jobs = self._draw_jobs(rng)
            opt_weight, opt_jobs = solve_optimal(jobs)
            if len(opt_jobs) == len(jobs):
                continue  # everything fits, nothing to decide (cannot happen with validated configs)
            greedy = greedy_weight(jobs)
            candidate = (jobs, opt_weight, opt_jobs, greedy)
            if cfg.require_greedy_gap and greedy == opt_weight:
                fallback = candidate
                continue
            chosen = candidate
            break
        if chosen is None:
            if fallback is None:
                raise ValueError(f"Failed to generate a non-trivial instance after {MAX_GENERATION_ATTEMPTS} attempts")
            chosen = fallback
        jobs, opt_weight, opt_jobs, greedy = chosen

        job_lines = "\n".join(f"{j['id']}: processing time {j['p']}, deadline {j['d']}, weight {j['w']}" for j in jobs)
        return {
            "question": QUESTION_TEMPLATE.format(jobs=job_lines),
            "answer": ", ".join(j["id"] for j in opt_jobs),
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "jobs": jobs,
                "optimal_weight": opt_weight,
                "optimal_schedule": [j["id"] for j in opt_jobs],
                "greedy_weight": greedy,
                "greedy_gap": greedy < opt_weight,
                "num_scheduled": len(opt_jobs),
                "partial_credit": cfg.partial_credit,
                "difficulty": {
                    "num_jobs": (cfg.min_jobs, cfg.max_jobs),
                    "processing_time": (cfg.min_processing_time, cfg.max_processing_time),
                    "weight": (cfg.min_weight, cfg.max_weight),
                    "tardiness_factor": cfg.tardiness_factor,
                    "due_date_range": cfg.due_date_range,
                },
            },
        }

    @staticmethod
    def _parse_schedule(answer: str) -> Optional[list[str]]:
        tokens = [t.strip().upper() for t in answer.strip().split(",")]
        if not tokens or any(_JOB_ID_RE.fullmatch(t) is None for t in tokens):
            return None
        return tokens

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """Score by simulating the proposed schedule: 1.0 if optimal, partial credit below 0.5 if feasible only."""
        if not isinstance(answer, str) or not answer.strip():
            return 0.0
        ids = self._parse_schedule(answer)
        if ids is None or len(set(ids)) != len(ids):
            return 0.0

        meta = entry["metadata"]
        by_id = {j["id"]: j for j in meta["jobs"]}
        if any(i not in by_id for i in ids):
            return 0.0

        on_time, weight = simulate([by_id[i] for i in ids])
        if not on_time:
            return 0.0
        optimum = meta["optimal_weight"]
        if weight >= optimum:
            return 1.0
        if meta.get("partial_credit", True):
            return 0.5 * weight / optimum
        return 0.0


class WeightedJobSchedulingCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(WeightedJobSchedulingCurriculum.__name__, WeightedJobSchedulingConfig)

        self._define_attributes(
            RangeAttributeDefinition(
                name="num_jobs",
                levels=[4, 6, 8, 10, 12, 15],
                description="Number of jobs in the instance",
                lower_field_name="min_jobs",
                upper_field_name="max_jobs",
                ensure_interval=True,
            ),
            ScalarAttributeDefinition(
                name="max_processing_time",
                field_name="max_processing_time",
                levels=[5, 10, 15, 20],
                description="Largest processing time of a job",
            ),
            ScalarAttributeDefinition(
                name="tardiness_factor",
                field_name="tardiness_factor",
                levels=[0.3, 0.5, 0.6, 0.7, 0.8],
                default_level=2,
                description="How early deadlines are placed; higher means fewer jobs fit",
            ),
        )


register_dataset(
    DATASET_NAME, WeightedJobSchedulingDataset, WeightedJobSchedulingConfig, WeightedJobSchedulingCurriculum
)
