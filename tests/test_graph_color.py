import json

import pytest

from reasoning_gym.algorithmic.graph_color import GraphColorConfig, GraphColorDataset


def test_graph_color():
    """Test basic properties and solution of generated items"""
    config = GraphColorConfig(seed=42, size=10, num_vertices=10, num_colors=4, edge_probability=0.4)
    dataset = GraphColorDataset(config)

    # easy
    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Test the scoring
        assert dataset.score_answer(answer=json.dumps(item["metadata"]["possible_answer"]), entry=item) == 1.0
        assert dataset.score_answer(answer=None, entry=item) == 0.0

    # medium
    config = GraphColorConfig(seed=42, size=1, num_vertices=10, num_colors=3, edge_probability=0.3)
    dataset = GraphColorDataset(config)

    for item in dataset:
        assert dataset.score_answer(answer=json.dumps(item["metadata"]["possible_answer"]), entry=item) == 1.0
        assert dataset.score_answer(answer=None, entry=item) == 0.0

    # hard
    config = GraphColorConfig(seed=42, size=1, num_vertices=40, num_colors=4, edge_probability=0.2)
    dataset = GraphColorDataset(config)

    for item in dataset:
        assert dataset.score_answer(answer=json.dumps(item["metadata"]["possible_answer"]), entry=item) == 1.0
        assert dataset.score_answer(answer=None, entry=item) == 0.0

    # v hard
    config = GraphColorConfig(seed=42, size=1, num_vertices=50, num_colors=3, edge_probability=0.1)
    dataset = GraphColorDataset(config)

    for item in dataset:
        assert dataset.score_answer(answer=json.dumps(item["metadata"]["possible_answer"]), entry=item) == 1.0
        assert dataset.score_answer(answer=None, entry=item) == 0.0
