import pytest

from reasoning_gym.logic.aiw import AliceInWonderlandConfig, AliceInWonderlandDataset, TaskType, OutputFormat

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
    
    with pytest.raises(AssertionError):
        config = AliceInWonderlandConfig(output_formats=[])  # No output formats
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
        assert "prompt" in item
        assert "right_answer" in item
        assert "description" in item
        
        # Verify answer is numeric and positive
        answer = int(item["right_answer"])
        assert answer > 0
        
        # Verify question contains at least one female name
        female_names = config.female_names
        assert any(name in item["prompt"] for name in female_names)

        # Verify question format
        if TaskType.SIBLINGS.value in item["description"]:
            assert any(phrase in item["prompt"] for phrase in ["brothers", "sisters"])
        elif TaskType.FRIENDS.value in item["description"]:
            assert "friends" in item["prompt"]
            
        # Verify output format
        if OutputFormat.RESTRICTED.value in item["description"]:
            assert "DO NOT OUTPUT ANY TEXT EXCEPT" in item["prompt"]
        elif OutputFormat.THINKING.value in item["description"]:
            assert "think carefully step by step" in item["prompt"]

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
        prompt = item["prompt"]
        numbers = [int(n) for n in prompt.split() if n.isdigit()]
        
        # Check all numbers are in reasonable range (1-6 as per implementation)
        assert all(1 <= n <= 12 for n in numbers), f"Numbers out of range: {numbers}"

def test_output_format_is_correct():
    """Test that the output format adheres to the user input"""
    config = AliceInWonderlandConfig(size=30, seed=42, output_formats=[OutputFormat.THINKING])
    dataset = AliceInWonderlandDataset(config)

    for item in dataset:
        prompt = item["prompt"]
        assert "think carefully step by step" in item["prompt"]
