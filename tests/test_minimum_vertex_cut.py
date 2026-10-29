import json

import pytest

from reasoning_gym import create_dataset
from reasoning_gym.graphs.minimum_vertex_cut import (
    MinimumVertexCutConfig,
    MinimumVertexCutDataset,
    _has_path,
    _minimum_vertex_cut,
)


def test_factory_registration():
    dataset = create_dataset("minimum_vertex_cut", size=2, seed=42)
    assert len(dataset) == 2
    assert dataset.score_answer(dataset[0]["answer"], dataset[0]) == 1.0


def test_generated_items_are_deterministic_and_well_formed():
    config = MinimumVertexCutConfig(seed=123, size=5)
    dataset_a = MinimumVertexCutDataset(config)
    dataset_b = MinimumVertexCutDataset(config)

    for idx in range(5):
        first = dataset_a[idx]
        second = dataset_b[idx]
        assert first == second
        assert set(first) == {"question", "answer", "metadata"}
        assert first["metadata"]["source_dataset"] == "minimum_vertex_cut"
        assert dataset_a.score_answer(first["answer"], first) == 1.0


def test_graph_is_connected_without_direct_terminal_edge_and_cut_is_optimal():
    dataset = MinimumVertexCutDataset(MinimumVertexCutConfig(seed=7, size=10))
    for idx in range(len(dataset)):
        entry = dataset[idx]
        graph = entry["metadata"]["graph"]
        vertices, edges = graph["vertices"], graph["edges"]
        source, target = graph["source"], graph["target"]
        edge_sets = {frozenset(edge) for edge in edges}

        assert frozenset((source, target)) not in edge_sets
        assert _has_path(vertices, edges, source, target, set())
        actual_cut = _minimum_vertex_cut(vertices, edges, source, target)
        assert dataset.config.min_cut_size <= len(actual_cut) <= dataset.config.max_cut_size
        assert not _has_path(vertices, edges, source, target, set(actual_cut))
        assert entry["metadata"]["solution"]["cut_size"] == len(actual_cut)


def test_scorer_accepts_an_alternative_optimum():
    dataset = MinimumVertexCutDataset(MinimumVertexCutConfig(seed=9, size=1))
    entry = dataset[0]
    # Two internally vertex-disjoint paths: s-a-b-t and s-c-d-t.
    graph = {
        "vertices": ["s", "t", "a", "b", "c", "d"],
        "edges": [["s", "a"], ["a", "b"], ["b", "t"], ["s", "c"], ["c", "d"], ["d", "t"]],
        "source": "s",
        "target": "t",
    }
    entry["metadata"]["graph"] = graph
    alternative = json.dumps({"cut_vertices": ["a", "c"], "cut_size": 2})
    assert dataset.score_answer(alternative, entry) == 1.0
    assert dataset.score_answer('```json\n' + alternative + '\n```', entry) == 1.0


def test_scorer_rejects_non_separating_cut_and_nonminimal_cut():
    dataset = MinimumVertexCutDataset(MinimumVertexCutConfig(seed=11, size=1))
    entry = dataset[0]
    entry["metadata"]["graph"] = {
        "vertices": ["s", "t", "a", "b"],
        "edges": [["s", "a"], ["a", "b"], ["b", "t"]],
        "source": "s",
        "target": "t",
    }

    assert dataset.score_answer('{"cut_vertices":["a"],"cut_size":1}', entry) == 1.0
    assert dataset.score_answer('{"cut_vertices":["s"],"cut_size":1}', entry) == 0.0
    assert dataset.score_answer('{"cut_vertices":["a","b"],"cut_size":2}', entry) == 0.0
    assert dataset.score_answer('{"cut_vertices":["a","a"],"cut_size":2}', entry) == 0.0
    assert dataset.score_answer("not json", entry) == 0.0
    assert dataset.score_answer(None, entry) == 0.0


def test_config_validation():
    with pytest.raises(AssertionError):
        MinimumVertexCutConfig(min_num_vertices=3).validate()
    with pytest.raises(AssertionError):
        MinimumVertexCutConfig(min_cut_size=0).validate()
    with pytest.raises(AssertionError):
        MinimumVertexCutConfig(min_cut_size=2, edge_probability=0.0).validate()
    with pytest.raises(AssertionError):
        MinimumVertexCutConfig(min_num_vertices=5, max_num_vertices=5, max_cut_size=4).validate()
