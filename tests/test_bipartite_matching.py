"""Tests for maximum bipartite matching generation and verification."""

import json
from itertools import product

import pytest

import reasoning_gym
from reasoning_gym.factory import create_curriculum, get_score_answer_fn
from reasoning_gym.graphs.bipartite_matching import (
    BipartiteMatchingConfig,
    BipartiteMatchingCurriculum,
    BipartiteMatchingDataset,
)


def brute_force_size(adjacency):
    """Independent exhaustive oracle: try skipping or assigning each worker."""

    def search(worker, used_jobs):
        if worker == len(adjacency):
            return 0
        best = search(worker + 1, used_jobs)
        for job in adjacency[worker]:
            if job not in used_jobs:
                best = max(best, 1 + search(worker + 1, used_jobs | {job}))
        return best

    return search(0, set())


def matching_entry(adjacency, num_jobs, matching_size):
    """Build a scoring fixture without using the implementation's solver."""
    return {
        "answer": "",
        "metadata": {
            "num_workers": len(adjacency),
            "num_jobs": num_jobs,
            "adjacency": adjacency,
            "matching_size": matching_size,
        },
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_num_workers": 0},
        {"min_num_workers": -1},
        {"min_num_workers": 9, "max_num_workers": 8},
        {"max_num_workers": 0},
        {"min_num_jobs": 0},
        {"min_num_jobs": -1},
        {"min_num_jobs": 9, "max_num_jobs": 8},
        {"max_num_jobs": 0},
        {"min_num_workers": 1.5},
        {"max_num_workers": 8.5},
        {"min_num_jobs": True},
        {"max_num_jobs": 8.5},
        {"edge_probability": -0.01},
        {"edge_probability": 1.01},
        {"edge_probability": float("nan")},
        {"size": 0},
        {"size": -1},
        {"size": 1.5},
        {"size": True},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(AssertionError):
        BipartiteMatchingDataset(BipartiteMatchingConfig(**kwargs))


def test_determinism_and_iteration():
    config = BipartiteMatchingConfig(seed=42, size=20)
    first = BipartiteMatchingDataset(config)
    second = BipartiteMatchingDataset(config)
    items = list(first)
    assert len(first) == len(items) == config.size
    assert items == list(first) == list(second)
    assert first[7] == items[7]  # Random access does not depend on iteration order.
    assert items != list(BipartiteMatchingDataset(BipartiteMatchingConfig(seed=43, size=20)))


@pytest.mark.parametrize("probability", [0.0, 0.2, 0.6, 1.0])
def test_generated_problems_against_brute_force(probability):
    config = BipartiteMatchingConfig(
        min_num_workers=1,
        max_num_workers=5,
        min_num_jobs=1,
        max_num_jobs=5,
        edge_probability=probability,
        seed=42,
        size=30,
    )
    dataset = BipartiteMatchingDataset(config)
    for idx, entry in enumerate(dataset):
        assert isinstance(entry["question"], str) and entry["question"]
        assert isinstance(entry["answer"], str)
        metadata = entry["metadata"]
        num_workers, num_jobs = metadata["num_workers"], metadata["num_jobs"]
        assert 1 <= num_workers <= 5 and 1 <= num_jobs <= 5
        assert metadata["source_dataset"] == "bipartite_matching"
        assert metadata["source_index"] == idx
        assert len(metadata["adjacency"]) == num_workers
        for worker, jobs in enumerate(metadata["adjacency"]):
            assert jobs == sorted(set(jobs))
            assert all(0 <= job < num_jobs for job in jobs)
            rendered = ", ".join(map(str, jobs)) if jobs else "none"
            assert f"Worker {worker}: {rendered}\n" in entry["question"]
        assert metadata["difficulty"]["edge_probability"] == probability
        assert metadata["matching_size"] == brute_force_size(metadata["adjacency"])
        assert json.loads(entry["answer"]) == metadata["solution"]
        assert dataset.score_answer(entry["answer"], entry) == 1.0
        # Entries also work after serialization for external evaluation.
        restored = json.loads(json.dumps(entry))
        assert dataset.score_answer(restored["answer"], restored) == 1.0


@pytest.mark.parametrize(
    "adjacency,num_jobs,expected",
    [
        ([[0]], 1, 1),
        ([[]], 1, 0),
        ([[], [], []], 2, 0),
        ([[0, 1], [0]], 2, 2),  # Greedy selection needs reassignment.
        ([[0], [0], [1, 2]], 3, 2),  # No perfect matching, despite equal group sizes.
        ([[0, 1, 2]], 3, 1),
        ([[0], [0], [0]], 1, 1),
        ([[0], [], [2], []], 4, 2),
        ([[0, 1], [0, 1]], 2, 2),
    ],
)
def test_solver_examples(adjacency, num_jobs, expected):
    dataset = BipartiteMatchingDataset(BipartiteMatchingConfig(seed=1))
    pairs = dataset._maximum_matching(adjacency)
    assert len(pairs) == expected
    assert dataset.score_answer(json.dumps(pairs), matching_entry(adjacency, num_jobs, expected)) == 1.0


def test_all_three_by_three_graphs():
    dataset = BipartiteMatchingDataset(BipartiteMatchingConfig(seed=1))
    for edges in product((False, True), repeat=9):
        adjacency = [[job for job in range(3) if edges[3 * worker + job]] for worker in range(3)]
        optimum = brute_force_size(adjacency)
        pairs = dataset._maximum_matching(adjacency)
        assert len(pairs) == optimum
        assert dataset.score_answer(json.dumps(pairs), matching_entry(adjacency, 3, optimum)) == 1.0


def test_long_augmenting_path():
    # The final worker forces a path longer than Python's default recursion limit.
    count = 1200
    adjacency = [[job, job + 1] for job in range(count - 1)] + [[0]]
    dataset = BipartiteMatchingDataset(BipartiteMatchingConfig(seed=1))
    pairs = dataset._maximum_matching(adjacency)
    assert pairs == [[worker, worker + 1] for worker in range(count - 1)] + [[count - 1, 0]]


@pytest.mark.parametrize("num_workers,num_jobs", [(1, 1), (2, 7), (7, 2), (8, 8)])
@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_empty_and_complete_graphs(num_workers, num_jobs, probability):
    dataset = BipartiteMatchingDataset(
        BipartiteMatchingConfig(
            min_num_workers=num_workers,
            max_num_workers=num_workers,
            min_num_jobs=num_jobs,
            max_num_jobs=num_jobs,
            edge_probability=probability,
            seed=42,
        )
    )
    entry = dataset[0]
    expected = min(num_workers, num_jobs) if probability else 0
    assert entry["metadata"]["matching_size"] == expected
    assert all(len(jobs) == (num_jobs if probability else 0) for jobs in entry["metadata"]["adjacency"])
    assert dataset.score_answer(entry["answer"], entry) == 1.0


@pytest.mark.parametrize("answer", ["[[0, 0], [1, 1]]", "[[0, 1], [1, 0]]", "  [ [1, 0],\n[0, 1] ]  "])
def test_alternative_maximum_matchings(answer):
    dataset = BipartiteMatchingDataset(BipartiteMatchingConfig(seed=1))
    entry = matching_entry([[0, 1], [0, 1]], 2, 2)
    assert dataset.score_answer(answer, entry) == 1.0


@pytest.mark.parametrize(
    "answer",
    [
        None,
        2,
        [[0, 0], [1, 1]],
        "",
        " ",
        "not JSON",
        '"[[0, 0], [1, 1]]"',
        "null",
        "true",
        "2",
        "{}",
        '{"0": 0, "1": 1}',
        "[]",
        "[[0, 0]]",  # Valid but not maximum.
        "[[0, 0], [0, 0]]",  # Duplicate pair.
        "[[0, 0], [0, 1]]",  # Reused worker.
        "[[0, 0], [1, 0]]",  # Reused job.
        "[[-1, 0], [1, 1]]",
        "[[0, -1], [1, 1]]",
        "[[2, 0], [1, 1]]",
        "[[0, 2], [1, 1]]",
        "[[0.0, 0], [1, 1]]",
        "[[0, 0.0], [1, 1]]",
        "[[false, 0], [1, 1]]",
        "[[0, false], [1, 1]]",
        '[["0", 0], [1, 1]]',
        "[[0, 0, 1], [1, 1]]",
        "[[0], [1, 1]]",
        "[0, [1, 1]]",
        "[null, [1, 1]]",
        "[[0, 0], [1, 1], [2, 2]]",
        "Answer: [[0, 0], [1, 1]]",
    ],
)
def test_invalid_answers(answer):
    dataset = BipartiteMatchingDataset(BipartiteMatchingConfig(seed=1))
    entry = matching_entry([[0, 1], [0, 1]], 2, 2)
    assert dataset.score_answer(answer, entry) == 0.0


def test_incompatible_and_nonmaximum_answers():
    dataset = BipartiteMatchingDataset(BipartiteMatchingConfig(seed=1))
    entry = matching_entry([[0, 1], [0]], 2, 2)
    assert dataset.score_answer("[[0, 1], [1, 0]]", entry) == 1.0
    assert dataset.score_answer("[[0, 0], [1, 1]]", entry) == 0.0
    # This is maximal (no pair can be added), but it is not maximum.
    assert dataset.score_answer("[[0, 0]]", entry) == 0.0


def test_empty_matching_scoring():
    dataset = BipartiteMatchingDataset(BipartiteMatchingConfig(seed=1))
    entry = matching_entry([[], []], 3, 0)
    assert dataset.score_answer("[]", entry) == 1.0
    assert dataset.score_answer(" [ ] ", entry) == 1.0
    assert dataset.score_answer("[[0, 0]]", entry) == 0.0
    assert dataset.score_answer("", entry) == 0.0


def test_factory_registration():
    dataset = reasoning_gym.create_dataset("bipartite_matching", seed=42, size=3)
    assert isinstance(dataset, BipartiteMatchingDataset)
    assert dataset.category == "graphs"
    assert isinstance(create_curriculum("bipartite_matching"), BipartiteMatchingCurriculum)
    entry = dataset[0]
    assert get_score_answer_fn("bipartite_matching")(entry["answer"], entry) == 1.0


def test_curriculum():
    curriculum = BipartiteMatchingCurriculum()
    defaults = {"seed": 42, "size": 10, "edge_probability": 0.4}
    config = curriculum.generate_configuration(defaults)
    assert config.min_num_workers == config.max_num_workers == 4
    assert config.min_num_jobs == config.max_num_jobs == 4
    assert config.seed == 42 and config.size == 10 and config.edge_probability == 0.4
    assert defaults == {"seed": 42, "size": 10, "edge_probability": 0.4}

    curriculum.increment_attr_level("num_workers")
    config = curriculum.generate_configuration(defaults)
    assert (config.min_num_workers, config.max_num_workers) == (4, 8)
    assert (config.min_num_jobs, config.max_num_jobs) == (4, 4)

    curriculum.increment_attr_level("num_jobs")
    curriculum.increment_attr_level("num_jobs")
    config = curriculum.generate_configuration(defaults)
    assert (config.min_num_jobs, config.max_num_jobs) == (4, 12)

    curriculum.decrement_attr_level("num_workers")
    config = curriculum.generate_configuration(defaults)
    assert config.min_num_workers == config.max_num_workers == 4
    assert (config.min_num_jobs, config.max_num_jobs) == (4, 12)
    assert len(BipartiteMatchingDataset(config)) == 10
