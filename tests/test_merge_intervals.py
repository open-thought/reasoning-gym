"""Tests for Merge Intervals questions generation"""

import pytest

import reasoning_gym
from reasoning_gym.algorithmic.merge_intervals import (
    MergeIntervalsConfig,
    MergeIntervalsCurriculum,
    MergeIntervalsDataset,
)

merge = MergeIntervalsDataset._merge_intervals


def test_merge_intervals_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        MergeIntervalsConfig(min_intervals=0).validate()

    with pytest.raises(AssertionError):
        MergeIntervalsConfig(min_intervals=5, max_intervals=3).validate()

    with pytest.raises(AssertionError):
        MergeIntervalsConfig(max_interval_length=-1).validate()

    with pytest.raises(AssertionError):
        MergeIntervalsConfig(max_value=5, max_interval_length=10).validate()


@pytest.mark.parametrize(
    "intervals, expected",
    [
        ([[1, 3], [2, 6], [8, 10], [15, 18]], [[1, 6], [8, 10], [15, 18]]),  # LeetCode example
        ([[1, 4], [4, 5]], [[1, 5]]),  # touching endpoints merge
        ([[1, 2], [3, 4]], [[1, 2], [3, 4]]),  # adjacent integers but a gap: no merge
        ([[1, 10], [2, 3], [4, 5]], [[1, 10]]),  # nested intervals
        ([[5, 7], [1, 2]], [[1, 2], [5, 7]]),  # unsorted input
        ([[3, 3], [3, 3]], [[3, 3]]),  # duplicate zero-length intervals
        ([[0, 0]], [[0, 0]]),  # single interval
    ],
)
def test_merge_intervals_known_cases(intervals, expected):
    original = [list(iv) for iv in intervals]
    assert merge(intervals) == expected
    assert intervals == original, "_merge_intervals must not modify its input"


def _covered(intervals: list[list[int]], x: float) -> bool:
    return any(s <= x <= e for s, e in intervals)


def test_merge_intervals_matches_brute_force():
    """Merged output must be sorted, disjoint, and cover exactly the same points as the input.

    Points are checked at every half-integer, so gaps like [1, 2] [3, 4] are detected.
    """
    dataset = MergeIntervalsDataset(MergeIntervalsConfig(max_intervals=25, max_value=60, size=200, seed=7))
    for item in dataset:
        intervals = item["metadata"]["intervals"]
        merged = item["metadata"]["solution"]

        for s, e in merged:
            assert s <= e
        for (_, e1), (s2, _) in zip(merged, merged[1:]):
            assert e1 < s2, "merged intervals must be sorted and must not touch or overlap"

        hi = max(e for _, e in intervals) + 1
        for twice_x in range(-2, 2 * hi + 1):
            x = twice_x / 2
            assert _covered(intervals, x) == _covered(merged, x)


def test_merge_intervals_dataset_deterministic():
    """Test that dataset generates same items with same seed"""
    config = MergeIntervalsConfig(seed=42, size=10)
    dataset1 = MergeIntervalsDataset(config)
    dataset2 = MergeIntervalsDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_merge_intervals_dataset_items():
    """Test basic properties of generated items"""
    config = MergeIntervalsConfig(
        min_intervals=3, max_intervals=8, max_value=40, max_interval_length=6, size=50, seed=42
    )
    dataset = MergeIntervalsDataset(config)

    for item in dataset:
        assert set(item.keys()) == {"question", "answer", "metadata"}
        assert isinstance(item["answer"], str)

        intervals = item["metadata"]["intervals"]
        assert config.min_intervals <= len(intervals) <= config.max_intervals
        for s, e in intervals:
            assert 0 <= s <= e <= config.max_value
            assert e - s <= config.max_interval_length

        assert str(intervals) in item["question"]
        assert item["metadata"]["num_merged"] == len(item["metadata"]["solution"])


def test_merge_intervals_has_variety():
    """Items should not be trivial: some merging happens, but not everything collapses."""
    dataset = MergeIntervalsDataset(MergeIntervalsConfig(min_intervals=5, max_intervals=10, size=100, seed=3))
    merged_counts = [item["metadata"]["num_merged"] for item in dataset]
    num_intervals = [item["metadata"]["num_intervals"] for item in dataset]

    assert any(m < n for m, n in zip(merged_counts, num_intervals)), "no item ever merges anything"
    assert sum(m == 1 for m in merged_counts) < len(merged_counts) / 2, "most items collapse to one interval"


def test_merge_intervals_score_answer():
    dataset = MergeIntervalsDataset(MergeIntervalsConfig(size=20, seed=1))
    for entry in dataset:
        assert dataset.score_answer(entry["answer"], entry) == 1.0
        assert dataset.score_answer(None, entry) == 0.0
        assert dataset.score_answer("", entry) == 0.0
        assert dataset.score_answer("not a list", entry) == 0.0
        assert dataset.score_answer("__import__('os').system('echo pwned')", entry) == 0.0

    entry = {"answer": "[[1, 6], [8, 10]]", "metadata": {"solution": [[1, 6], [8, 10]]}}
    assert dataset.score_answer("[[1,6],[8,10]]", entry) == 1.0  # whitespace-insensitive
    assert dataset.score_answer("[[1, 6], [8, 11]]", entry) < 1.0
    assert dataset.score_answer("[[1, 10]]", entry) < 1.0
    assert dataset.score_answer("((1, 6), (8, 10))", entry) == 1.0  # tuples are fine
    assert dataset.score_answer("[[8, 10], [1, 6]]", entry) == 0.5  # right intervals, not sorted
    assert dataset.score_answer("[[1, 6], [8, 10], [8, 10]]", entry) == 0.0  # duplicates are wrong
    assert dataset.score_answer("[[1, 6, 7]]", entry) == 0.0
    assert dataset.score_answer("[[True, 6], [8, 10]]", entry) == 0.0
    assert dataset.score_answer("[[1.0, 6], [8, 10]]", entry) == 0.0


def test_merge_intervals_generates_touching_case():
    """The generator should regularly produce intervals that only touch at an endpoint."""
    dataset = MergeIntervalsDataset(MergeIntervalsConfig(size=100, seed=11))
    touching = 0
    for item in dataset:
        intervals = item["metadata"]["intervals"]
        ends = {e for _, e in intervals}
        touching += any(s in ends for s, _ in intervals)
    assert touching >= 20


def test_merge_intervals_registered():
    dataset = reasoning_gym.create_dataset("merge_intervals", size=5, seed=0)
    for entry in dataset:
        assert dataset.score_answer(entry["answer"], entry) == 1.0


def test_merge_intervals_curriculum():
    curriculum = MergeIntervalsCurriculum()
    base_value = {"size": 150, "seed": 1}

    base_cfg: MergeIntervalsConfig = curriculum.generate_configuration(base_value)
    assert base_cfg.seed == 1
    assert base_cfg.size == 150
    assert base_cfg.min_intervals == 5 and base_cfg.max_intervals == 5
    assert base_cfg.max_value == 50

    curriculum.increment_attr_level("num_intervals")
    curriculum.increment_attr_level("max_value")
    increased_cfg = curriculum.generate_configuration(base_value)
    assert increased_cfg.min_intervals == 5 and increased_cfg.max_intervals == 10
    assert increased_cfg.max_value == 100

    curriculum.decrement_attr_level("num_intervals")
    decreased_cfg = curriculum.generate_configuration(base_value)
    assert decreased_cfg.min_intervals == 5 and decreased_cfg.max_intervals == 5
