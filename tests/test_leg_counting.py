"""Tests for leg counting task generation"""

import pytest

from reasoning_gym.arithmetic.leg_counting import ANIMALS, LegCountingConfig, LegCountingDataset


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
