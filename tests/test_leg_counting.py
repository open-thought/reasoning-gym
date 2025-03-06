"""Tests for leg counting task generation"""

import pytest

from reasoning_gym.arithmetic.leg_counting import ANIMALS, LegCountingConfig, LegCountingCurriculum, LegCountingDataset


def test_leg_counting_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = LegCountingConfig(min_animals=0)
        config.validate()

    with pytest.raises(AssertionError):
        config = LegCountingConfig(min_animals=10, max_animals=5)
        config.validate()

    with pytest.raises(AssertionError):
        config = LegCountingConfig(max_instances=0)
        config.validate()


def test_leg_counting_dataset_deterministic():
    """Test that dataset generates same items with same seed"""
    config = LegCountingConfig(seed=42, size=10)
    dataset1 = LegCountingDataset(config)
    dataset2 = LegCountingDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_leg_counting_dataset_items():
    """Test basic properties of generated items"""
    config = LegCountingConfig(min_animals=2, max_animals=4, max_instances=2, size=10, seed=42)
    dataset = LegCountingDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        # Check item structure
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata
        assert "animals" in item["metadata"]
        assert "total_legs" in item["metadata"]

        # Verify animal count constraints
        animals = item["metadata"]["animals"]
        assert len(animals) >= config.min_animals
        assert len(animals) <= config.max_animals

        # Verify instance count constraints
        assert all(1 <= count <= config.max_instances for count in animals.values())

        # Verify leg counting is correct
        total_legs = sum(count * ANIMALS[animal] for animal, count in animals.items())
        assert str(total_legs) == item["answer"]
        assert total_legs == item["metadata"]["total_legs"]


def test_leg_counting_dataset_iteration():
    """Test that iteration respects dataset size"""
    config = LegCountingConfig(size=5, seed=42)
    dataset = LegCountingDataset(config)

    items = list(dataset)
    assert len(items) == config.size

    # Test multiple iterations yield same items
    assert items == list(dataset)


def test_leg_counting_animal_validation():
    """Test that all animals have valid leg counts"""
    # Verify all animals have non-negative leg counts
    assert all(legs >= 0 for legs in ANIMALS.values())

    # Verify common animals have expected leg counts
    assert ANIMALS["spider"] == 8
    assert ANIMALS["insect"] == 6
    assert ANIMALS["dog"] == 4
    assert ANIMALS["chicken"] == 2
    assert ANIMALS["snake"] == 0


def test_leg_counting_curriculum():
    curriculum = LegCountingCurriculum()

    base_value = {"size": 150, "seed": 1}

    base_cfg: LegCountingConfig = curriculum.generate_configuration(base_value)
    assert base_cfg.seed == 1
    assert base_cfg.size == 150
    assert base_cfg.min_animals == 10 and base_cfg.max_animals == 10
    assert base_cfg.min_instances == 10 and base_cfg.max_instances == 10

    # test incrementing attribute levels
    curriculum.increment_attr_level("num_animals")
    curriculum.increment_attr_level("num_instances")
    increased_cfg = curriculum.generate_configuration(base_value)
    assert increased_cfg.min_animals == 10 and increased_cfg.max_animals == 50
    assert increased_cfg.min_instances == 10 and increased_cfg.max_instances == 100

    # test decrementing attribute level for num_animals again
    curriculum.decrement_attr_level("num_animals")
    partially_decreased_cfg = curriculum.generate_configuration(base_value)
    assert partially_decreased_cfg.min_animals == 10 and partially_decreased_cfg.max_animals == 10
    assert partially_decreased_cfg.min_instances == 10 and partially_decreased_cfg.max_instances == 100
