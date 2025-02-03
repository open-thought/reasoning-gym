import pytest

from reasoning_gym.cognition import Arc1DConfig, Arc1DDataset


def test_arc_1d_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = Arc1DConfig(min_size=0)
        config.validate()

    with pytest.raises(AssertionError):
        config = Arc1DConfig(min_size=30, max_size=20)
        config.validate()

    with pytest.raises(AssertionError):
        config = Arc1DConfig(num_train=0)
        config.validate()


def test_arc_1d_deterministic():
    """Test that dataset generates same items with same seed"""
    config = Arc1DConfig(seed=42, size=10)
    dataset1 = Arc1DDataset(config)
    dataset2 = Arc1DDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_arc_1d_items():
    """Test basic properties of generated items"""
    config = Arc1DConfig(min_size=10, max_size=15, num_train=2, size=50, seed=42)
    dataset = Arc1DDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata contents
        metadata = item["metadata"]
        assert "task_name" in metadata
        assert "size" in metadata
        assert "train_examples" in metadata
        assert "test_example" in metadata

        # Verify size constraints
        assert config.min_size <= metadata["size"] <= config.max_size

        # Check training examples
        train_examples = metadata["train_examples"]
        assert len(train_examples) == config.num_train
        for example in train_examples:
            assert "input" in example
            assert "output" in example
            assert len(example["input"]) == metadata["size"]
            assert len(example["output"]) == metadata["size"]

        # Check test example
        test_example = metadata["test_example"]
        assert "input" in test_example
        assert "output" in test_example
        assert len(test_example["input"]) == metadata["size"]
        assert len(test_example["output"]) == metadata["size"]


def test_arc_1d_iteration():
    """Test that iteration respects dataset size"""
    config = Arc1DConfig(size=5, seed=42)  # Small size for testing
    dataset = Arc1DDataset(config)

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


def test_arc_1d_scoring():
    """Test answer scoring logic"""
    config = Arc1DConfig(size=1, seed=42)
    dataset = Arc1DDataset(config)
    entry = dataset[0]

    # Test exact match
    assert dataset.score_answer(entry["answer"], entry) == 1.0

    # Test partial match (answer contained within response)
    assert dataset.score_answer(f"The answer is: {entry['answer']}", entry) == 0.5

    # Test incorrect answer
    assert dataset.score_answer("wrong answer", entry) == 0.01

    # Test None answer
    assert dataset.score_answer(None, entry) == 0.0
