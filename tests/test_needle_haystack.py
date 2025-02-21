import pytest

from reasoning_gym.cognition.needle_haystack import NeedleHaystackConfig, NeedleHaystackDataset


def test_needle_haystack():
    """Test basic properties and solution of generated items"""
    config = NeedleHaystackConfig(seed=42, size=50, num_statements=50)
    dataset = NeedleHaystackDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Test the scoring
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0
        assert dataset.score_answer(answer="david bowie rules", entry=item) == 0.01
        assert dataset.score_answer(answer=None, entry=item) == 0.0

    config = NeedleHaystackConfig(seed=42, size=1, num_statements=500)
    dataset = NeedleHaystackDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Test the scoring
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0
        assert dataset.score_answer(answer=None, entry=item) == 0.0

    config = NeedleHaystackConfig(seed=42, size=1, num_statements=5000)
    dataset = NeedleHaystackDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Test the scoring
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0
        assert dataset.score_answer(answer=None, entry=item) == 0.0

    config = NeedleHaystackConfig(seed=42, size=1, num_statements=50000)
    dataset = NeedleHaystackDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Test the scoring
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0
        assert dataset.score_answer(answer=None, entry=item) == 0.0

    config = NeedleHaystackConfig(seed=42, size=1, num_statements=500000)
    dataset = NeedleHaystackDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Test the scoring
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0
        assert dataset.score_answer(answer=None, entry=item) == 0.0
