"""Tests for minimum spanning tree questions generation"""

import random
import string
from itertools import combinations

import pytest

from reasoning_gym.graphs import (
    MinimumSpanningTreeConfig,
    MinimumSpanningTreeCurriculum,
    MinimumSpanningTreeDataset,
)
from reasoning_gym.graphs.minimum_spanning_tree import generate_graph, parse_edges

# Small graphs with narrow weights: brute force is cheap and weight ties are common
SMALL = dict(min_nodes=3, max_nodes=6, min_extra_edges=0, max_extra_edges=4, min_weight=1, max_weight=4)


# ---- independent helpers (deliberately do not reuse the code under test) ----


def is_connected(num_nodes, edges):
    nodes = string.ascii_uppercase[:num_nodes]
    reached = {nodes[0]}
    changed = True
    while changed:
        changed = False
        for edge in edges:
            a, b = edge[0], edge[1]
            if (a in reached) != (b in reached):
                reached.update((a, b))
                changed = True
    return len(reached) == num_nodes


def is_spanning_tree(num_nodes, edges):
    return len(edges) == num_nodes - 1 and is_connected(num_nodes, edges)


def all_spanning_trees(num_nodes, edges):
    return [c for c in combinations(edges, num_nodes - 1) if is_spanning_tree(num_nodes, c)]


def total_weight(tree):
    return sum(edge[2] for edge in tree)


def as_text(tree, rng):
    """Write a tree with random edge order and random node order inside each edge."""
    pairs = [(e[1], e[0]) if rng.random() < 0.5 else (e[0], e[1]) for e in tree]
    rng.shuffle(pairs)
    return ", ".join(f"{a}-{b}" for a, b in pairs)


# ---- configuration ----


def test_minimum_spanning_tree_config_validation():
    """Test that invalid configs raise appropriate errors"""
    invalid_configs = [
        dict(min_nodes=1),
        dict(min_nodes=5, max_nodes=4),
        dict(max_nodes=27),
        dict(min_extra_edges=-1),
        dict(min_extra_edges=3, max_extra_edges=2),
        dict(min_weight=0),
        dict(min_weight=5, max_weight=4),
        dict(size=0),
    ]
    for kwargs in invalid_configs:
        with pytest.raises(AssertionError):
            MinimumSpanningTreeConfig(**kwargs).validate()


# ---- generation ----


def test_minimum_spanning_tree_deterministic():
    """Test that dataset generates same items with same seed"""
    config = MinimumSpanningTreeConfig(seed=42, size=20)
    dataset1 = MinimumSpanningTreeDataset(config)
    dataset2 = MinimumSpanningTreeDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_minimum_spanning_tree_different_seeds():
    """Test that different seeds give different items"""
    items1 = list(MinimumSpanningTreeDataset(MinimumSpanningTreeConfig(seed=1, size=10)))
    items2 = list(MinimumSpanningTreeDataset(MinimumSpanningTreeConfig(seed=2, size=10)))
    assert items1 != items2


def test_minimum_spanning_tree_items():
    """Test basic properties of generated items"""
    config = MinimumSpanningTreeConfig(
        min_nodes=3, max_nodes=7, min_extra_edges=1, max_extra_edges=5, min_weight=2, max_weight=9, size=100, seed=42
    )
    dataset = MinimumSpanningTreeDataset(config)

    for i, item in enumerate(dataset):
        assert isinstance(item, dict)
        assert isinstance(item["question"], str)
        assert isinstance(item["answer"], str)

        meta = item["metadata"]
        assert meta["source_dataset"] == "minimum_spanning_tree"
        assert meta["source_index"] == i
        assert meta["difficulty"] == {"num_nodes": (3, 7), "num_extra_edges": (1, 5)}

        num_nodes = meta["num_nodes"]
        edges = [tuple(e) for e in meta["edges"]]
        assert 3 <= num_nodes <= 7
        assert is_connected(num_nodes, edges)
        assert num_nodes - 1 <= len(edges) <= min(num_nodes - 1 + 5, num_nodes * (num_nodes - 1) // 2)
        for a, b, w in edges:
            assert a < b
            assert 2 <= w <= 9
            assert f"{a} - {b} (weight {w})" in item["question"]

        # the stored answer is a spanning tree whose weight equals the stored optimum
        weights = {(a, b): w for a, b, w in edges}
        tree = [(a, b, weights[(a, b)]) for a, b in parse_edges(item["answer"])]
        assert is_spanning_tree(num_nodes, tree)
        assert total_weight(tree) == meta["mst_weight"]
        assert dataset.score_answer(item["answer"], item) == 1.0


def test_generate_graph_is_valid():
    """Graphs must be connected, free of duplicate edges, and have the expected edge count"""
    for seed in range(500):
        rng = random.Random(seed)
        num_nodes = rng.randint(2, 26)
        extra = rng.randint(0, 40)
        edges = generate_graph(rng, num_nodes, extra, 1, 20)
        pairs = [(a, b) for a, b, _ in edges]

        assert is_connected(num_nodes, edges)
        assert len(pairs) == len(set(pairs))
        assert all(a < b for a, b in pairs)
        assert len(edges) == min(num_nodes - 1 + extra, num_nodes * (num_nodes - 1) // 2)
        assert all(1 <= w <= 20 for _, _, w in edges)


def test_minimum_spanning_tree_extra_edges_are_capped():
    """Asking for more extra edges than can exist yields the complete graph"""
    config = MinimumSpanningTreeConfig(
        min_nodes=4, max_nodes=4, min_extra_edges=100, max_extra_edges=100, size=10, seed=1
    )
    for item in MinimumSpanningTreeDataset(config):
        assert len(item["metadata"]["edges"]) == 6  # 4 nodes -> 6 possible edges


def test_minimum_spanning_tree_two_nodes():
    """The smallest graph has a single edge, which is also the answer"""
    config = MinimumSpanningTreeConfig(min_nodes=2, max_nodes=2, min_extra_edges=0, max_extra_edges=3, size=5, seed=1)
    dataset = MinimumSpanningTreeDataset(config)
    for item in dataset:
        assert item["answer"] == "A-B"
        assert dataset.score_answer("B-A", item) == 1.0
        assert dataset.score_answer("", item) == 0.0


def test_minimum_spanning_tree_iteration():
    """Test that iteration respects dataset size"""
    config = MinimumSpanningTreeConfig(size=5, seed=42)
    dataset = MinimumSpanningTreeDataset(config)

    assert len(list(dataset)) == config.size
    assert list(dataset) == list(dataset)  # repeated iteration gives the same items


# ---- reference answer and scoring, checked against brute force ----


def test_minimum_spanning_tree_reference_answer_is_optimal():
    """Kruskal's answer must match the best of ALL spanning trees found by brute force"""
    dataset = MinimumSpanningTreeDataset(MinimumSpanningTreeConfig(**SMALL, size=150, seed=7))
    for item in dataset:
        meta = item["metadata"]
        edges = [tuple(e) for e in meta["edges"]]
        trees = all_spanning_trees(meta["num_nodes"], edges)
        assert meta["mst_weight"] == min(total_weight(t) for t in trees)


def test_minimum_spanning_tree_score_accepts_every_optimal_tree():
    """Any minimum-weight tree is correct, whatever the edge order or node order"""
    dataset = MinimumSpanningTreeDataset(MinimumSpanningTreeConfig(**SMALL, size=150, seed=42))
    rng = random.Random(0)
    graphs_with_ties = 0

    for item in dataset:
        meta = item["metadata"]
        edges = [tuple(e) for e in meta["edges"]]
        trees = all_spanning_trees(meta["num_nodes"], edges)
        optimal = [t for t in trees if total_weight(t) == meta["mst_weight"]]
        for tree in optimal:
            assert dataset.score_answer(as_text(tree, rng), item) == 1.0
        graphs_with_ties += len(optimal) > 1

    assert graphs_with_ties > 0, "test data never contained weight ties, so alternatives were not exercised"


def test_minimum_spanning_tree_score_rejects_suboptimal_trees():
    """A valid spanning tree that is not minimal must score 0"""
    dataset = MinimumSpanningTreeDataset(MinimumSpanningTreeConfig(**SMALL, size=150, seed=42))
    rng = random.Random(1)
    rejected = 0

    for item in dataset:
        meta = item["metadata"]
        edges = [tuple(e) for e in meta["edges"]]
        for tree in all_spanning_trees(meta["num_nodes"], edges):
            if total_weight(tree) > meta["mst_weight"]:
                assert dataset.score_answer(as_text(tree, rng), item) == 0.0
                rejected += 1

    assert rejected > 0


def test_minimum_spanning_tree_score_rejects_malformed_answers():
    dataset = MinimumSpanningTreeDataset(MinimumSpanningTreeConfig(**SMALL, size=20, seed=3))
    item = dataset[0]
    assert item["metadata"]["num_nodes"] >= 3
    pairs = item["answer"].split(", ")

    assert dataset.score_answer(item["answer"], item) == 1.0
    assert dataset.score_answer(None, item) == 0.0
    assert dataset.score_answer("", item) == 0.0
    assert dataset.score_answer("   ", item) == 0.0
    assert dataset.score_answer("I do not know", item) == 0.0
    assert dataset.score_answer(", ".join(pairs[:-1]), item) == 0.0  # one edge too few
    assert dataset.score_answer(", ".join(pairs + [pairs[0]]), item) == 0.0  # repeated edge, one too many
    assert dataset.score_answer(", ".join(pairs[:-1] + [pairs[0]]), item) == 0.0  # right count, but a duplicate
    assert dataset.score_answer(", ".join(pairs[:-1] + ["A-A"]), item) == 0.0  # self loop
    assert dataset.score_answer(", ".join(pairs[:-1] + ["A-Z"]), item) == 0.0  # edge not in the graph


def test_minimum_spanning_tree_score_rejects_cycles():
    """n-1 existing edges that contain a cycle leave a node unconnected, so they are not a tree"""
    config = MinimumSpanningTreeConfig(min_nodes=4, max_nodes=4, min_extra_edges=0, max_extra_edges=2, size=200, seed=5)
    dataset = MinimumSpanningTreeDataset(config)
    tested = 0

    for item in dataset:
        present = {(a, b) for a, b, _ in item["metadata"]["edges"]}
        for x, y, z in combinations("ABCD", 3):
            if {(x, y), (x, z), (y, z)} <= present:
                # three edges forming a triangle: 3 = n-1 edges, but node outside the triangle is cut off
                assert dataset.score_answer(f"{x}-{y}, {x}-{z}, {y}-{z}", item) == 0.0
                tested += 1

    assert tested > 0, "no triangle found in the test data"


def test_parse_edges():
    assert parse_edges("A-B, C-D") == [("A", "B"), ("C", "D")]
    assert parse_edges("B-A") == [("A", "B")]  # node order inside an edge is normalised
    assert parse_edges("A - B ; B - C") == [("A", "B"), ("B", "C")]  # spacing and separators
    assert parse_edges("A\u2013B\nC\u2014D") == [("A", "B"), ("C", "D")]  # en dash and em dash
    assert parse_edges("no edges here") == []


# ---- curriculum ----


def test_minimum_spanning_tree_curriculum():
    curriculum = MinimumSpanningTreeCurriculum()

    base_value = {"size": 150, "seed": 1}

    base_cfg: MinimumSpanningTreeConfig = curriculum.generate_configuration(base_value)
    assert base_cfg.seed == 1
    assert base_cfg.size == 150
    assert base_cfg.min_nodes == 4 and base_cfg.max_nodes == 4
    assert base_cfg.min_extra_edges == 1 and base_cfg.max_extra_edges == 1

    # test incrementing attribute levels for num_nodes & num_extra_edges attributes
    curriculum.increment_attr_level("num_nodes")
    curriculum.increment_attr_level("num_extra_edges")

    increased_cfg = curriculum.generate_configuration(base_value)
    assert increased_cfg.min_nodes == 4 and increased_cfg.max_nodes == 6
    assert increased_cfg.min_extra_edges == 1 and increased_cfg.max_extra_edges == 2
