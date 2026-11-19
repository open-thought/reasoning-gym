"""Tests for the weighted job scheduling task"""

from itertools import combinations
from random import Random

import pytest

from reasoning_gym.algorithmic.weighted_job_scheduling import (
    WeightedJobSchedulingConfig,
    WeightedJobSchedulingCurriculum,
    WeightedJobSchedulingDataset,
    greedy_weight,
    simulate,
    solve_optimal,
)


def brute_force_optimum(jobs: list[dict]) -> int:
    """Try every subset, run it in deadline order, keep the best on-time weight."""
    best = 0
    for r in range(len(jobs) + 1):
        for subset in combinations(jobs, r):
            ordered = sorted(subset, key=lambda j: (j["d"], j["id"]))
            ok, weight = simulate(ordered)
            if ok:
                best = max(best, weight)
    return best


def random_jobs(rng: Random, n: int) -> list[dict]:
    return [
        {"id": f"J{i + 1}", "p": rng.randint(1, 8), "d": rng.randint(1, 30), "w": rng.randint(1, 20)} for i in range(n)
    ]


def make_entry(jobs: list[dict], partial_credit: bool = True) -> dict:
    best, chosen = solve_optimal(jobs)
    return {
        "answer": ", ".join(j["id"] for j in chosen),
        "metadata": {"jobs": jobs, "optimal_weight": best, "partial_credit": partial_credit},
    }


def test_config_validation():
    bad_configs = [
        {"min_jobs": 1},
        {"min_jobs": 5, "max_jobs": 4},
        {"min_processing_time": 0},
        {"min_processing_time": 5, "max_processing_time": 4},
        {"min_weight": 0},
        {"min_weight": 5, "max_weight": 4},
        {"tardiness_factor": 0.0},
        {"tardiness_factor": 1.0},
        {"due_date_range": 1.0},
        {"tardiness_factor": 0.1, "due_date_range": 0.4},
        {"size": 0},
    ]
    for overrides in bad_configs:
        with pytest.raises(AssertionError):
            WeightedJobSchedulingConfig(**overrides).validate()
    WeightedJobSchedulingConfig().validate()


def test_deterministic():
    cfg = WeightedJobSchedulingConfig(seed=42, size=10)
    a = WeightedJobSchedulingDataset(cfg)
    b = WeightedJobSchedulingDataset(cfg)
    other = WeightedJobSchedulingDataset(WeightedJobSchedulingConfig(seed=43, size=10))
    for i in range(10):
        assert a[i] == b[i]
    assert any(a[i] != other[i] for i in range(10))


def test_item_structure():
    ds = WeightedJobSchedulingDataset(WeightedJobSchedulingConfig(seed=1, size=20))
    for item in ds:
        assert isinstance(item["question"], str) and isinstance(item["answer"], str)
        meta = item["metadata"]
        for key in (
            "source_dataset",
            "source_index",
            "jobs",
            "optimal_weight",
            "optimal_schedule",
            "greedy_weight",
            "greedy_gap",
            "num_scheduled",
            "partial_credit",
            "difficulty",
        ):
            assert key in meta
        assert meta["source_dataset"] == "weighted_job_scheduling"
        for job in meta["jobs"]:
            assert job["id"] in item["question"]
            assert job["d"] >= job["p"]
        assert ds.config.min_jobs <= len(meta["jobs"]) <= ds.config.max_jobs


def test_solver_matches_brute_force():
    rng = Random(2024)
    for _ in range(300):
        jobs = random_jobs(rng, rng.randint(1, 10))
        best, chosen = solve_optimal(jobs)
        assert best == brute_force_optimum(jobs)
        ok, weight = simulate(chosen)
        assert ok and weight == best


def test_reference_answer_scores_one():
    ds = WeightedJobSchedulingDataset(WeightedJobSchedulingConfig(seed=5, size=100))
    for item in ds:
        assert ds.score_answer(item["answer"], item) == 1.0


def test_other_optimal_orderings_score_one():
    # J1 and J2 both fit in either order; J3 is too heavy to fit alongside them
    jobs = [
        {"id": "J1", "p": 2, "d": 10, "w": 5},
        {"id": "J2", "p": 3, "d": 10, "w": 5},
        {"id": "J3", "p": 9, "d": 10, "w": 4},
    ]
    entry = make_entry(jobs)
    ds = WeightedJobSchedulingDataset(WeightedJobSchedulingConfig(seed=1, size=1))
    assert entry["metadata"]["optimal_weight"] == 10
    assert ds.score_answer("J1, J2", entry) == 1.0
    assert ds.score_answer("J2, J1", entry) == 1.0


def test_late_schedule_scores_zero():
    jobs = [
        {"id": "J1", "p": 5, "d": 5, "w": 10},
        {"id": "J2", "p": 5, "d": 6, "w": 10},
    ]
    entry = make_entry(jobs)
    ds = WeightedJobSchedulingDataset(WeightedJobSchedulingConfig(seed=1, size=1))
    assert ds.score_answer("J1, J2", entry) == 0.0  # J2 would end at 10 > 6
    assert ds.score_answer("J2, J1", entry) == 0.0  # J1 ends at 10 > 5


def test_invalid_answers_score_zero():
    ds = WeightedJobSchedulingDataset(WeightedJobSchedulingConfig(seed=1, size=1))
    item = ds[0]
    first = item["metadata"]["optimal_schedule"][0]
    for bad in [
        None,
        "",
        "   ",
        "banana",
        "J99",
        f"{first}, {first}",
        f"{first}, J99",
        f"{first}, ",
        f"{first} then J2",
        f"[{first}]",
        "1, 2",
    ]:
        assert ds.score_answer(bad, item) == 0.0, bad


def test_partial_credit():
    jobs = [
        {"id": "J1", "p": 5, "d": 5, "w": 12},
        {"id": "J2", "p": 5, "d": 5, "w": 8},
        {"id": "J3", "p": 5, "d": 5, "w": 3},
    ]
    ds = WeightedJobSchedulingDataset(WeightedJobSchedulingConfig(seed=1, size=1))
    with_credit = make_entry(jobs, partial_credit=True)
    assert with_credit["answer"] == "J1"
    assert ds.score_answer("J2", with_credit) == pytest.approx(0.5 * 8 / 12)
    assert 0 < ds.score_answer("J3", with_credit) < 0.5
    assert ds.score_answer("J1", with_credit) == 1.0

    no_credit = make_entry(jobs, partial_credit=False)
    assert ds.score_answer("J2", no_credit) == 0.0
    assert ds.score_answer("J1", no_credit) == 1.0


def test_format_tolerance():
    ds = WeightedJobSchedulingDataset(WeightedJobSchedulingConfig(seed=3, size=10))
    for item in ds:
        ids = item["metadata"]["optimal_schedule"]
        assert ds.score_answer(",".join(ids), item) == 1.0
        assert ds.score_answer(" , ".join(i.lower() for i in ids), item) == 1.0
        assert ds.score_answer("  " + ", ".join(ids) + "\n", item) == 1.0


def test_deadlines_are_tight():
    ds = WeightedJobSchedulingDataset(WeightedJobSchedulingConfig(seed=11, size=300))
    for item in ds:
        meta = item["metadata"]
        assert meta["num_scheduled"] < len(meta["jobs"])
        assert meta["num_scheduled"] >= 1


def test_higher_tardiness_schedules_fewer_jobs():
    def mean_fraction(tf: float) -> float:
        ds = WeightedJobSchedulingDataset(
            WeightedJobSchedulingConfig(seed=8, size=200, min_jobs=8, max_jobs=8, tardiness_factor=tf)
        )
        return sum(x["metadata"]["num_scheduled"] / len(x["metadata"]["jobs"]) for x in ds) / len(ds)

    assert mean_fraction(0.8) < mean_fraction(0.5) < mean_fraction(0.3)


def test_greedy_gap():
    ds = WeightedJobSchedulingDataset(WeightedJobSchedulingConfig(seed=21, size=200, require_greedy_gap=True))
    gaps = [x["metadata"]["greedy_gap"] for x in ds]
    assert sum(gaps) / len(gaps) > 0.95
    for item in ds:
        meta = item["metadata"]
        assert meta["greedy_weight"] == greedy_weight(meta["jobs"])
        assert meta["greedy_weight"] <= meta["optimal_weight"]
        if meta["greedy_gap"]:
            assert meta["greedy_weight"] < meta["optimal_weight"]

    free = WeightedJobSchedulingDataset(WeightedJobSchedulingConfig(seed=21, size=200, require_greedy_gap=False))
    assert any(not x["metadata"]["greedy_gap"] for x in free)


def test_curriculum():
    curriculum = WeightedJobSchedulingCurriculum()
    base = curriculum.generate_configuration({"seed": 1})
    assert isinstance(base, WeightedJobSchedulingConfig)
    base.validate()

    for attr in curriculum.attributes:
        while curriculum.increment_attr_level(attr):
            cfg = curriculum.generate_configuration({"seed": 1, "size": 5})
            cfg.validate()
            ds = WeightedJobSchedulingDataset(cfg)
            for item in ds:
                assert ds.score_answer(item["answer"], item) == 1.0

    top = curriculum.generate_configuration({"seed": 1})
    assert top.max_jobs > base.max_jobs
    assert top.max_processing_time > base.max_processing_time
    assert top.tardiness_factor > base.tardiness_factor
    assert top.max_jobs == 15 and top.max_processing_time == 20 and top.tardiness_factor == 0.8
