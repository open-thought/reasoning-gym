import pytest

from reasoning_gym.geometry.simple_geometry import SimpleGeometryConfig, SimpleGeometryDataset


def test_simple_geometry_config_validation():
    """Test invalid configs raise appropriate errors."""
    # min_sides < 3
    with pytest.raises(AssertionError):
        config = SimpleGeometryConfig(min_sides=2, max_sides=5)
        config.validate()

    # max_sides < min_sides
    with pytest.raises(AssertionError):
        config = SimpleGeometryConfig(min_sides=4, max_sides=3)
        config.validate()

    # Invalid angles
    with pytest.raises(AssertionError):
        config = SimpleGeometryConfig(min_angle=-10)
        config.validate()

    with pytest.raises(AssertionError):
        config = SimpleGeometryConfig(min_angle=10, max_angle=5)
        config.validate()


def test_simple_geometry_dataset_deterministic():
    """Test the dataset generates the same items with the same seed."""
    config = SimpleGeometryConfig(seed=42, size=5, min_sides=3, max_sides=4)
    dataset1 = SimpleGeometryDataset(config)
    dataset2 = SimpleGeometryDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i], (
            f"Item mismatch at index {i} for same seed. " f"Dataset1: {dataset1[i]} vs Dataset2: {dataset2[i]}"
        )


def test_simple_geometry_dataset_items():
    """Test basic properties of generated items."""
    config = SimpleGeometryConfig(min_sides=3, max_sides=5, min_angle=10, max_angle=120, size=10, seed=123)
    dataset = SimpleGeometryDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        # Check structure
        assert isinstance(item, dict), "Generated item must be a dictionary."
        assert "question" in item, "Item must contain a 'question' key."
        assert "answer" in item, "Item must contain an 'answer' key."
        assert "metadata" in item, "Item must contain a 'metadata' key."

        metadata = item["metadata"]
        assert "n_sides" in metadata, "Metadata should contain 'n_sides'."
        assert "missing_angle_rounded" in metadata, "Metadata should contain the computed 'missing_angle_rounded'."

        # Check that the missing angle is a valid float or integer
        missing_angle = float(item["answer"])
        assert missing_angle > 0, f"Missing angle should be positive, found {missing_angle}"


def test_simple_geometry_dataset_iteration():
    """Test that iteration respects dataset size and is repeatable."""
    config = SimpleGeometryConfig(min_sides=3, max_sides=4, size=5, seed=42)
    dataset = SimpleGeometryDataset(config)

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
