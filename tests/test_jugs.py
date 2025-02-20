import json

import pytest

from reasoning_gym.algorithmic.jugs import JugsConfig, JugsDataset


def test_jugs():
    """Test basic properties and solution of generated items"""
    config = JugsConfig(seed=42, size=10, num_jugs=3, difficulty=20)
    dataset = JugsDataset(config)

    # easy
    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Test the scoring
        assert dataset.score_answer(answer=json.dumps(item["metadata"]["possible_answer"]), entry=item) == 1.0
        assert dataset.score_answer(answer=None, entry=item) == 0.0
