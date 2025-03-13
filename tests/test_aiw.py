import pytest

from reasoning_gym.logic.aiw import (
    AliceInWonderlandConfig,
    AliceInWonderlandCurriculum,
    AliceInWonderlandDataset,
    TaskType,
)


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


def test_aiw_curriculum():
    """Test the AIW curriculum functionality"""
    curriculum = AliceInWonderlandCurriculum()

    base_value = {"size": 100, "seed": 42}

    # Test default configuration
    base_cfg: AliceInWonderlandConfig = curriculum.generate_configuration(base_value)
    assert base_cfg.seed == 42
    assert base_cfg.size == 100
    assert base_cfg.min_entities == 4 and base_cfg.max_entities == 4
    assert base_cfg.task_type_weights == [1.0, 0.0, 0.0]  # Default is siblings only

    # Test incrementing task_type_weight attribute
    curriculum.increment_attr_level("task_type_weight")
    task_weight_cfg = curriculum.generate_configuration(base_value)
    assert task_weight_cfg.task_type_weights == [0.9, 0.05, 0.05]  # Second level adds some friends/colleagues

    # Test incrementing num_entities attribute
    curriculum.increment_attr_level("num_entities")
    entities_cfg = curriculum.generate_configuration(base_value)
    assert entities_cfg.max_entities == 6  # Increased max entities
    assert entities_cfg.task_type_weights == [0.9, 0.05, 0.05]  # Should preserve task weight level

    # Test decrementing task_type_weight attribute
    curriculum.decrement_attr_level("task_type_weight")
    updated_cfg = curriculum.generate_configuration(base_value)
    assert updated_cfg.task_type_weights == [1.0, 0.0, 0.0]  # Back to default weights
    assert updated_cfg.max_entities == 6  # Should preserve entities level

    # Test global level setting
    curriculum.set_global_level(2)  # Set all attributes to level 2
    global_cfg = curriculum.generate_configuration(base_value)
    assert global_cfg.task_type_weights == [0.7, 0.15, 0.15]  # Level 2 of task weights
    assert global_cfg.max_entities == 8  # Level 2 of num_entities
