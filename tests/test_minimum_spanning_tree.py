"""Tests for Minimum Spanning Tree problem generation"""

from itertools import combinations

import pytest

import reasoning_gym
from reasoning_gym.graphs.minimum_spanning_tree import (
    MinimumSpanningTreeConfig,
    MinimumSpanningTreeCurriculum,
    MinimumSpanningTreeDataset,
)


def _is_spanning_tree(num_nodes: int, edges: list[list[int]]) -> bool:
    """Check that the edges connect all nodes without cycles"""
    if len(edges) != num_nodes - 1:
        return False
    parent = list(range(num_nodes))

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    for u, v, _ in edges:
        ru, rv = find(u), find(v)
        if ru == rv:
            return False
        parent[ru] = rv
    return True


def _brute_force_mst_weight(num_nodes: int, edges: list[list[int]]) -> int:
    """Minimum weight over all spanning trees (only feasible for small graphs)"""
    best = None
    for subset in combinations(edges, num_nodes - 1):
        if _is_spanning_tree(num_nodes, list(subset)):
            weight = sum(w for _, _, w in subset)
            best = weight if best is None else min(best, weight)
    return best


def test_minimum_spanning_tree_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = MinimumSpanningTreeConfig(min_nodes=1)  # Must be >= 2
        config.validate()

    with pytest.raises(AssertionError):
        config = MinimumSpanningTreeConfig(min_nodes=6, max_nodes=4)  # min > max
        config.validate()

    with pytest.raises(AssertionError):
        config = MinimumSpanningTreeConfig(min_weight=0)  # Must be >= 1
        config.validate()

    with pytest.raises(AssertionError):
        config = MinimumSpanningTreeConfig(min_weight=10, max_weight=5)  # min > max
        config.validate()

    with pytest.raises(AssertionError):
        config = MinimumSpanningTreeConfig(p_extra_edge=1.5)  # Must be in [0, 1]
        config.validate()


def test_minimum_spanning_tree_dataset_deterministic():
    """Test that dataset generates same items with same seed"""
    config = MinimumSpanningTreeConfig(seed=42, size=10)
    dataset1 = MinimumSpanningTreeDataset(config)
    dataset2 = MinimumSpanningTreeDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_minimum_spanning_tree_dataset_items():
    """Test basic properties of generated items"""
    config = MinimumSpanningTreeConfig(min_nodes=3, max_nodes=10, min_weight=2, max_weight=30, size=50, seed=42)
    dataset = MinimumSpanningTreeDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        metadata = item["metadata"]
        assert metadata["source_dataset"] == "minimum_spanning_tree"
        assert metadata["source_index"] == i
        assert "difficulty" in metadata

        num_nodes = metadata["num_nodes"]
        assert config.min_nodes <= num_nodes <= config.max_nodes

        # Edges are unique, within range, and listed in the question
        seen = set()
        for u, v, w in metadata["edges"]:
            assert 0 <= u < num_nodes and 0 <= v < num_nodes and u != v
            assert config.min_weight <= w <= config.max_weight
            assert (min(u, v), max(u, v)) not in seen
            seen.add((min(u, v), max(u, v)))
            assert f"{u} - {v}: {w}" in item["question"]

        # Oracle tree is a valid spanning tree whose weight matches the answer
        assert _is_spanning_tree(num_nodes, metadata["mst_edges"])
        assert sum(w for _, _, w in metadata["mst_edges"]) == int(item["answer"]) == metadata["solution"]


def test_minimum_spanning_tree_answer_is_optimal():
    """Test that the oracle answer matches a brute-force search over all spanning trees"""
    config = MinimumSpanningTreeConfig(min_nodes=3, max_nodes=6, p_extra_edge=0.5, size=30, seed=7)
    dataset = MinimumSpanningTreeDataset(config)

    for item in dataset:
        metadata = item["metadata"]
        assert _brute_force_mst_weight(metadata["num_nodes"], metadata["edges"]) == int(item["answer"])


def test_minimum_spanning_tree_dataset_iteration():
    """Test that iteration respects dataset size"""
    config = MinimumSpanningTreeConfig(size=5, seed=42)
    dataset = MinimumSpanningTreeDataset(config)

    items = list(dataset)
    assert len(items) == config.size

    # Test multiple iterations yield same items
    assert items == list(dataset)


def test_minimum_spanning_tree_score_answer():
    """Test the score_answer method"""
    config = MinimumSpanningTreeConfig(seed=42, size=5)
    dataset = MinimumSpanningTreeDataset(config)
    item = dataset[0]
    oracle = item["answer"]

    assert dataset.score_answer(oracle, item) == 1.0
    assert dataset.score_answer(f"  {oracle}\n", item) == 1.0
    assert dataset.score_answer(str(int(oracle) + 1), item) == 0.0
    assert dataset.score_answer("not a number", item) == 0.0
    assert dataset.score_answer("", item) == 0.0
    assert dataset.score_answer(None, item) == 0.0


def test_minimum_spanning_tree_registered():
    """Test that the dataset can be created through the registry"""
    dataset = reasoning_gym.create_dataset("minimum_spanning_tree", size=3, seed=1)
    assert len(list(dataset)) == 3


def test_minimum_spanning_tree_curriculum():
    """Test curriculum creates valid configs at various levels"""
    curriculum = MinimumSpanningTreeCurriculum()

    base_value = {"size": 150, "seed": 1}

    base_cfg: MinimumSpanningTreeConfig = curriculum.generate_configuration(base_value)
    assert base_cfg.seed == 1
    assert base_cfg.size == 150
    assert base_cfg.min_nodes == 4 and base_cfg.max_nodes == 4
    assert base_cfg.max_weight == 10
    assert base_cfg.p_extra_edge == 0.2

    # Increment attributes
    curriculum.increment_attr_level("nodes")
    curriculum.increment_attr_level("max_weight")
    curriculum.increment_attr_level("p_extra_edge")
    increased_cfg = curriculum.generate_configuration(base_value)
    assert increased_cfg.min_nodes == 4 and increased_cfg.max_nodes == 8
    assert increased_cfg.max_weight == 20
    assert increased_cfg.p_extra_edge == 0.4

    # Decrement nodes back
    curriculum.decrement_attr_level("nodes")
    partial_cfg = curriculum.generate_configuration(base_value)
    assert partial_cfg.min_nodes == 4 and partial_cfg.max_nodes == 4
    assert partial_cfg.max_weight == 20
    assert partial_cfg.p_extra_edge == 0.4
