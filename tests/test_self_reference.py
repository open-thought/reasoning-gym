import pytest

from reasoning_gym.logic.self_reference import SelfReferenceConfig, SelfReferenceDataset


def test_self_reference():
    """Test basic properties and solution of generated items"""

    # Easy
    config = SelfReferenceConfig(seed=42, size=20, difficulty=1)
    dataset = SelfReferenceDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Test the scoring
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0
        assert dataset.score_answer(answer=99, entry=item) == 0.1
        assert dataset.score_answer(answer="99", entry=item) == 0.1
        assert dataset.score_answer(answer=None, entry=item) == 0.0

    # # Medium
    config = SelfReferenceConfig(seed=42, size=1, difficulty=5)
    dataset = SelfReferenceDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Test the scoring
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0
        assert dataset.score_answer(answer=99, entry=item) == 0.1
        assert dataset.score_answer(answer="99", entry=item) == 0.1
        assert dataset.score_answer(answer=None, entry=item) == 0.0

    # # Hard
    config = SelfReferenceConfig(seed=42, size=1, difficulty=10)
    dataset = SelfReferenceDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Test the scoring
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0
        assert dataset.score_answer(answer=99, entry=item) == 0.1
        assert dataset.score_answer(answer="99", entry=item) == 0.1
        assert dataset.score_answer(answer=None, entry=item) == 0.0
