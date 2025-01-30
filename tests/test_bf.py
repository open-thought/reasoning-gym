import pytest

from reasoning_gym.code.bf import BFConfig, BFDataset


def test_bf():
    """Test basic properties and solution of generated items"""

    # Easy
    config = BFConfig(seed=42, size=20, difficulty=1)
    dataset = BFDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata contains required fields
        assert "bfit_code" in item["metadata"]
        assert "bf_program" in item["metadata"]

        # Test the scoring
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0
        assert dataset.score_answer(answer=None, entry=item) == 0.0
        assert dataset.score_answer(answer="Love is a battlefield", entry=item) == 0.01

    # Medium
    config = BFConfig(seed=43, size=20, difficulty=2)
    dataset = BFDataset(config)
    for item in dataset:
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0

    # Hard
    config = BFConfig(seed=44, size=20, difficulty=3)
    dataset = BFDataset(config)
    for item in dataset:
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0
