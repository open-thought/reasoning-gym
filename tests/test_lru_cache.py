"""LRU semantics, independent oracle checks, generation, and reward validation."""

import itertools
import json
from dataclasses import replace

import pytest

from reasoning_gym import create_dataset, get_score_answer_fn
from reasoning_gym.algorithmic import LRUCacheConfig, LRUCacheCurriculum, LRUCacheDataset
from reasoning_gym.factory import create_curriculum


def timestamp_oracle(accesses, capacity):
    """Independent model: evict the resident with the oldest access timestamp."""
    last_seen = {}
    trace = ""
    for clock, item in enumerate(accesses):
        trace += "H" if item in last_seen else "M"
        if item not in last_seen and len(last_seen) == capacity:
            del last_seen[min(last_seen, key=last_seen.get)]
        last_seen[item] = clock
    return {"trace": trace, "cache": sorted(last_seen, key=last_seen.get)}


@pytest.mark.parametrize(
    "changes",
    [
        {"min_capacity": 0},
        {"min_capacity": 6, "max_capacity": 5},
        {"max_num_accesses": 0},
        {"min_num_items": -1},
        {"min_num_items": 13, "max_num_items": 12},
        {"min_num_accesses": 31, "max_num_accesses": 30},
        {"min_capacity": True},
        {"max_num_items": 3.5},
        {"hit_probability": -0.01},
        {"hit_probability": 1.01},
        {"hit_probability": float("nan")},
        {"size": 0},
        {"size": -1},
    ],
)
def test_invalid_configuration(changes):
    with pytest.raises(AssertionError):
        LRUCacheDataset(replace(LRUCacheConfig(), **changes))


@pytest.mark.parametrize(
    "accesses,capacity,expected",
    [
        (list("ABCADB"), 3, {"trace": "MMMHMM", "cache": list("ADB")}),
        (list("AAAA"), 1, {"trace": "MHHH", "cache": ["A"]}),
        (list("ABBA"), 1, {"trace": "MMHM", "cache": ["A"]}),
        (list("ABAC"), 2, {"trace": "MMHM", "cache": list("AC")}),
        (list("ABC"), 5, {"trace": "MMM", "cache": list("ABC")}),
        (list("ABACA"), 2, {"trace": "MMHMH", "cache": list("CA")}),
    ],
)
def test_known_traces(accesses, capacity, expected):
    assert LRUCacheDataset._simulate(accesses, capacity) == expected


def test_exhaustive_small_traces():
    """Check every length 1..7 trace over three symbols at capacities 1..4."""
    for length in range(1, 8):
        for accesses in itertools.product("ABC", repeat=length):
            for capacity in range(1, 5):
                assert LRUCacheDataset._simulate(accesses, capacity) == timestamp_oracle(accesses, capacity)


@pytest.mark.parametrize("seed", [0, 42, 2026])
def test_generated_items(seed):
    dataset = LRUCacheDataset(LRUCacheConfig(seed=seed, size=50))
    twin = LRUCacheDataset(LRUCacheConfig(seed=seed, size=50))
    # Access order must not affect deterministic indexed generation.
    assert dataset[20] == twin[20]
    items = list(dataset)
    assert len(items) == 50
    assert items == list(twin) == list(dataset)
    for idx, entry in enumerate(items):
        metadata = entry["metadata"]
        capacity = metadata["capacity"]
        accesses = metadata["accesses"]
        assert 2 <= capacity <= 5
        assert 12 <= len(accesses) <= 30
        assert 6 <= metadata["num_items"] <= 12
        assert all(item in {f"P{i}" for i in range(metadata["num_items"])} for item in accesses)
        assert metadata["source_dataset"] == "lru_cache"
        assert metadata["source_index"] == idx
        assert " ".join(accesses) in entry["question"]
        assert f"at most {capacity} items" in entry["question"]
        expected = timestamp_oracle(accesses, capacity)
        assert json.loads(entry["answer"]) == metadata["solution"] == expected
        assert dataset.score_answer(entry["answer"], entry) == 1.0


@pytest.mark.parametrize(
    "capacity,num_items,length,probability",
    [(1, 1, 1, 0), (1, 1, 8, 1), (5, 2, 20, 0), (1, 4, 20, 0), (2, 5, 20, 1)],
)
def test_generation_boundaries(capacity, num_items, length, probability):
    dataset = LRUCacheDataset(
        LRUCacheConfig(
            min_capacity=capacity,
            max_capacity=capacity,
            min_num_items=num_items,
            max_num_items=num_items,
            min_num_accesses=length,
            max_num_accesses=length,
            hit_probability=probability,
            seed=42,
        )
    )
    for idx in range(10):
        entry = dataset[idx]
        assert json.loads(entry["answer"]) == timestamp_oracle(entry["metadata"]["accesses"], capacity)
        if num_items == 1 or probability == 1:
            assert json.loads(entry["answer"])["trace"] == "M" + "H" * (length - 1)
        elif probability == 0 and num_items > capacity:
            assert json.loads(entry["answer"])["trace"] == "M" * length


def test_default_distribution():
    dataset = LRUCacheDataset(LRUCacheConfig(seed=42, size=500))
    mixed = evicting = hits = accesses = 0
    for entry in dataset:
        metadata = entry["metadata"]
        trace = metadata["solution"]["trace"]
        mixed += "H" in trace and "M" in trace
        evicting += trace.count("M") > metadata["capacity"]
        hits += trace.count("H")
        accesses += len(trace)
    assert mixed >= 475
    assert evicting >= 400
    assert 0.30 < hits / accesses < 0.55


@pytest.fixture
def scoring_case():
    dataset = LRUCacheDataset(LRUCacheConfig(seed=42))
    return dataset, {"answer": '{"trace":"MMMHMM","cache":["A","D","B"]}'}


@pytest.mark.parametrize(
    "answer",
    [
        '{"trace":"MMMHMM","cache":["A","D","B"]}',
        '  { "cache": ["A", "D", "B"], "trace": "MMMHMM" }\n',
    ],
)
def test_scoring_accepts_equivalent_json(scoring_case, answer):
    dataset, entry = scoring_case
    assert dataset.score_answer(answer, entry) == 1.0


@pytest.mark.parametrize(
    "answer",
    [
        None,
        "",
        "MMMHMM",
        "null",
        "[]",
        "42",
        "true",
        '{"trace":"MMMMMM","cache":["A","D","B"]}',
        '{"trace":"MMMHMM","cache":["B","D","A"]}',
        '{"trace":"MMMHMM","cache":["A","D","B","B"]}',
        '{"trace":"MMMHMM"}',
        '{"trace":"MMMHMM","cache":["A","D","B"],"extra":0}',
        '{"trace":"MMMHMM","trace":"MMMHMM","cache":["A","D","B"]}',
        '{"trace":true,"cache":["A","D","B"]}',
        '{"trace":"MMMHMM","cache":"ADB"}',
        '{"trace":"MMMHMM","cache":[1,2,3]}',
        '{"trace":"M M M H M M","cache":["A","D","B"]}',
        '{"trace":"mmmhmm","cache":["A","D","B"]}',
        'Here is the answer: {"trace":"MMMHMM","cache":["A","D","B"]}',
        '```json\n{"trace":"MMMHMM","cache":["A","D","B"]}\n```',
        '{"trace":"MMMHMM","cache":["A","D","B"]} {}',
        '{"trace":"MMMHMM","cache":["A","D","B"],}',
        "[" * 2000,
        {"trace": "MMMHMM", "cache": ["A", "D", "B"]},
    ],
)
def test_scoring_rejects_invalid_or_incorrect_answers(scoring_case, answer):
    dataset, entry = scoring_case
    assert dataset.score_answer(answer, entry) == 0.0


def test_registry():
    dataset = create_dataset("lru_cache", seed=7, size=3)
    assert isinstance(dataset, LRUCacheDataset)
    assert dataset.category == "algorithmic"
    assert len(dataset) == 3
    entry = dataset[0]
    assert get_score_answer_fn("lru_cache")(entry["answer"], entry) == 1.0
    assert isinstance(create_curriculum("lru_cache"), LRUCacheCurriculum)


def test_curriculum():
    curriculum = LRUCacheCurriculum()
    config = curriculum.generate_configuration({"seed": 42, "size": 10})
    assert (config.min_capacity, config.max_capacity) == (2, 2)
    assert (config.min_num_accesses, config.max_num_accesses) == (12, 12)
    assert (config.min_num_items, config.max_num_items) == (6, 6)
    assert config.seed == 42 and config.size == 10
    curriculum.increment_attr_level("capacity")
    assert curriculum.generate_configuration().max_capacity == 4
    curriculum.decrement_attr_level("capacity")
    assert curriculum.generate_configuration().max_capacity == 2
    for level in range(4):
        curriculum.set_global_level(level)
        config = curriculum.generate_configuration({"seed": 42, "size": 3})
        config.validate()
        dataset = LRUCacheDataset(config)
        for entry in dataset:
            assert json.loads(entry["answer"]) == timestamp_oracle(
                entry["metadata"]["accesses"], entry["metadata"]["capacity"]
            )
