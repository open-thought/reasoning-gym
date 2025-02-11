from reasoning_gym.algorithmic.ab import ABConfig, ABDataset


def test_abconfig_puzzles():
    """Test basic properties and solution of generated items"""
    config = ABConfig(seed=42, size=10, length=5)
    dataset = ABDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Test the scoring
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0
        assert dataset.score_answer(answer=None, entry=item) == 0.0

    config = ABConfig(seed=42, size=10, length=15)
    dataset = ABDataset(config)
    for item in dataset:
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0
        assert dataset.score_answer(answer=None, entry=item) == 0.0

    config = ABConfig(seed=42, size=10, length=25)
    dataset = ABDataset(config)
    for item in dataset:
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0
        assert dataset.score_answer(answer=None, entry=item) == 0.0
