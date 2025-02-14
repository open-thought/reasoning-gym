import pytest

from reasoning_gym.algorithmic.game_of_life import GameOfLifeConfig, GameOfLifeDataset


def test_game_of_life_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = GameOfLifeConfig(grid_size_x=2)  # Too small
        config.validate()

    with pytest.raises(AssertionError):
        config = GameOfLifeConfig(grid_size_y=1000)  # Too large
        config.validate()

    with pytest.raises(AssertionError):
        config = GameOfLifeConfig(grid_size_x=5, grid_size_y=5, filled_cells=26)  # Too many cells
        config.validate()


def test_game_of_life_deterministic():
    """Test that dataset generates same items with same seed"""
    config = GameOfLifeConfig(seed=42, size=10)
    dataset1 = GameOfLifeDataset(config)
    dataset2 = GameOfLifeDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_game_of_life_basic_properties():
    """Test basic properties and solution of generated items"""
    config = GameOfLifeConfig(seed=42, size=10, grid_size_x=20, grid_size_y=20, filled_cells=200, simulation_steps=1)
    dataset = GameOfLifeDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata contains required fields
        assert "grid_size_x" in item["metadata"]
        assert "grid_size_y" in item["metadata"]
        assert "filled_cells" in item["metadata"]
        assert "simulation_steps" in item["metadata"]

        # Test the scoring
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0
        assert dataset.score_answer(answer=None, entry=item) == 0.0
        assert dataset.score_answer(answer="invalid json", entry=item) == 0.01


def test_game_of_life_iteration():
    """Test that iteration respects dataset size"""
    config = GameOfLifeConfig(size=5, seed=42)  # Small size for testing
    dataset = GameOfLifeDataset(config)

    # Test manual iteration
    items = []
    for item in dataset:
        items.append(item)
    assert len(items) == config.size, "Iterator should yield exactly size items"

    # Test list conversion
    items = list(dataset)
    assert len(items) == config.size, "Iterator should yield exactly size items"

    # Test multiple iterations
    first_items = list(dataset)
    second_items = list(dataset)
    assert first_items == second_items, "Multiple iterations should yield same items"
