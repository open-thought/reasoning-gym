import pytest

from reasoning_gym.games.countdown_game import CountdownGameConfig, CountdownGameDataset


def test_countdown_game_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = CountdownGameConfig(min_numbers=1)  # Too few numbers
        config.validate()

    with pytest.raises(AssertionError):
        config = CountdownGameConfig(min_numbers=4, max_numbers=3)  # Invalid range
        config.validate()

    with pytest.raises(AssertionError):
        config = CountdownGameConfig(operators=["^"])  # Invalid operator
        config.validate()


def test_countdown_game_deterministic():
    """Test that dataset generates same items with same seed"""
    config = CountdownGameConfig(seed=42, size=10)
    dataset1 = CountdownGameDataset(config)
    dataset2 = CountdownGameDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_countdown_game_items():
    """Test basic properties of generated items"""
    config = CountdownGameConfig(
        min_numbers=3,
        max_numbers=5,
        min_value=1,
        max_value=20,  # Small numbers for testing
        min_target=10,
        max_target=100,
        size=50,
        seed=42
    )
    dataset = CountdownGameDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item
        
        # Check metadata contains required fields
        assert "numbers" in item["metadata"]
        assert "target" in item["metadata"]
        assert "expression" in item["metadata"]
        
        # Verify number of source numbers is within config range
        assert config.min_numbers <= len(item["metadata"]["numbers"]) <= config.max_numbers
        
        # Verify target is within config range
        assert config.min_target <= item["metadata"]["target"] <= config.max_target
        
        # Verify all numbers are within config range
        assert all(config.min_value <= n <= config.max_value for n in item["metadata"]["numbers"])


def test_countdown_game_randomization():
    """Test number randomization configuration"""
    config = CountdownGameConfig(
        min_numbers=4,
        max_numbers=4,  # Fixed size for testing
        randomize_numbers=False,
        size=10,
        seed=42
    )
    
    # Without randomization, numbers should appear in same order
    dataset = CountdownGameDataset(config)
    first_item = dataset[0]
    expr_nums = [int(n) for n in first_item["metadata"]["expression"].replace("(","").replace(")","").split(" ") if n.isdigit()]
    assert expr_nums == first_item["metadata"]["numbers"]
    
    # With randomization, numbers might appear in different order
    config.randomize_numbers = True
    dataset = CountdownGameDataset(config)
    first_item = dataset[0]
    expr_nums = [int(n) for n in first_item["metadata"]["expression"].replace("(","").replace(")","").split(" ") if n.isdigit()]
    assert sorted(expr_nums) == sorted(first_item["metadata"]["numbers"])
