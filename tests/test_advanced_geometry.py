import pytest

from reasoning_gym.geometry.advanced_geometry import (
    AdvancedGeometryConfig,
    AdvancedGeometryCurriculum,
    AdvancedGeometryDataset,
)


def test_advanced_geometry_config_validation():
    """Test that invalid configs raise appropriate errors."""
    # min_coord >= max_coord
    with pytest.raises(AssertionError):
        config = AdvancedGeometryConfig(min_coord=5, max_coord=5)
        config.validate()

    with pytest.raises(AssertionError):
        config = AdvancedGeometryConfig(min_coord=10, max_coord=0)
        config.validate()

    # size <= 0
    with pytest.raises(AssertionError):
        config = AdvancedGeometryConfig(size=0)
        config.validate()

    # Empty task_types
    with pytest.raises(AssertionError):
        config = AdvancedGeometryConfig(task_types=[])
        config.validate()


def test_advanced_geometry_dataset_deterministic():
    """Test the dataset generates the same items with the same seed."""
    config = AdvancedGeometryConfig(min_coord=-5, max_coord=5, size=5, seed=42)
    dataset1 = AdvancedGeometryDataset(config)
    dataset2 = AdvancedGeometryDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i], (
            f"Item mismatch at index {i} for same seed. " f"Dataset1: {dataset1[i]} vs Dataset2: {dataset2[i]}"
        )


def test_advanced_geometry_dataset_items():
    """Test basic properties of generated items."""
    config = AdvancedGeometryConfig(min_coord=-3, max_coord=3, size=5, seed=123)
    dataset = AdvancedGeometryDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        # Check structure
        assert isinstance(item, dict), "Generated item must be a dictionary."
        assert "question" in item, "Item must contain a 'question' key."
        assert "answer" in item, "Item must contain an 'answer' key."
        assert "metadata" in item, "Item must contain a 'metadata' key."

        # Basic metadata checks
        metadata = item["metadata"]
        assert (
            "A" in metadata and "B" in metadata and "C" in metadata
        ), "Metadata should contain coordinates for points A, B, and C."

        # Check answer format depending on task type
        # For angle measure tasks, answer should end with '°'
        if "angle_measure" in item["question"].lower() or "angle at" in item["question"].lower():
            assert item["answer"].endswith("°"), f"Expected angle measure in degrees, got {item['answer']}"


def test_advanced_geometry_dataset_iteration():
    """Test that iteration respects dataset size and is repeatable."""
    config = AdvancedGeometryConfig(min_coord=-2, max_coord=2, size=3, seed=999)
    dataset = AdvancedGeometryDataset(config)

    # Test manual iteration
    items = []
    for item in dataset:
        items.append(item)
    assert len(items) == config.size, "Iterator should yield exactly 'size' items."

    # Test list conversion
    items_list = list(dataset)
    assert len(items_list) == config.size, "List conversion should yield exactly 'size' items."

    # Test multiple iterations produce the same results
    first_items = list(dataset)
    second_items = list(dataset)
    assert first_items == second_items, "Multiple iterations should yield the same items."


def test_advanced_geometry_curriculum():
    curriculum = AdvancedGeometryCurriculum()

    base_value = {"size": 150, "seed": 1}

    base_cfg: AdvancedGeometryConfig = curriculum.generate_configuration(base_value)
    assert base_cfg.seed == 1
    assert base_cfg.size == 150
    assert base_cfg.min_coord == -10
    assert base_cfg.max_coord == 10

    # test incrementing attribute levels
    curriculum.increment_attr_level("min_coord")
    curriculum.increment_attr_level("max_coord")
    increased_cfg = curriculum.generate_configuration(base_value)
    assert increased_cfg.min_coord == -100
    assert increased_cfg.max_coord == 100

    # test decrementing attribute level
    curriculum.decrement_attr_level("min_coord")
    partially_decreased_cfg = curriculum.generate_configuration(base_value)
    assert partially_decreased_cfg.min_coord == -10
    assert partially_decreased_cfg.max_coord == 100
