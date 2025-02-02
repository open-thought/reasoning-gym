import pytest

from reasoning_gym.logic.aiw import AliceInWonderlandConfig, AliceInWonderlandDataset, TaskType


def test_aiw_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = AliceInWonderlandConfig(male_names=[])  # Empty male names
        config.validate()

    with pytest.raises(AssertionError):
        config = AliceInWonderlandConfig(female_names=[])  # Empty female names
        config.validate()

    with pytest.raises(AssertionError):
        config = AliceInWonderlandConfig(female_names=["Mary", "Jane"])  # No Alice
        config.validate()

    with pytest.raises(AssertionError):
        config = AliceInWonderlandConfig(task_types=[])  # No task types
        config.validate()


def test_aiw_deterministic():
    """Test that dataset generates same items with same seed"""
    config = AliceInWonderlandConfig(seed=42, size=10)
    dataset1 = AliceInWonderlandDataset(config)
    dataset2 = AliceInWonderlandDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_aiw_items():
    """Test basic properties of generated items"""
    config = AliceInWonderlandConfig(size=50, seed=42)
    dataset = AliceInWonderlandDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Verify answer is numeric and positive
        answer = int(item["answer"])
        assert answer > 0

        # Verify question contains at least one female name
        female_names = config.female_names
        assert any(name in item["question"] for name in female_names)

        # Verify question task type characteristics
        task_type = item["metadata"]["task_type"]
        if task_type == TaskType.SIBLINGS.value:
            assert any(phrase in item["question"] for phrase in ["brothers", "sisters"])
        elif task_type == TaskType.FRIENDS.value:
            assert "friends" in item["question"]
        elif task_type == TaskType.COLLEAGUES:
            assert "colleagues" in item["question"]


def test_aiw_iteration():
    """Test that iteration works correctly"""
    config = AliceInWonderlandConfig(size=5, seed=42)
    dataset = AliceInWonderlandDataset(config)

    # Test manual iteration
    items = []
    for item in dataset:
        items.append(item)
    assert len(items) == config.size

    # Test list conversion
    items = list(dataset)
    assert len(items) == config.size

    # Test multiple iterations yield same results
    first_items = list(dataset)
    second_items = list(dataset)
    assert first_items == second_items


def test_aiw_random_ranges():
    """Test that generated numbers stay within expected ranges"""
    config = AliceInWonderlandConfig(size=30, seed=42, max_entities=12)
    dataset = AliceInWonderlandDataset(config)

    for item in dataset:
        question = item["question"]
        numbers = [int(n) for n in question.split() if n.isdigit()]

        # Check all numbers are in reasonable range (1-6 as per implementation)
        assert all(1 <= n <= 12 for n in numbers), f"Numbers out of range: {numbers}"
