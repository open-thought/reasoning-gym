import pytest

from reasoning_gym.cognition.number_sequences import NumberSequenceConfig, NumberSequenceDataset, Operation, PatternRule


def test_sequence_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = NumberSequenceConfig(min_terms=3)  # Too few terms
        config.validate()

    with pytest.raises(AssertionError):
        config = NumberSequenceConfig(min_terms=6, max_terms=5)
        config.validate()

    with pytest.raises(AssertionError):
        config = NumberSequenceConfig(min_value=100, max_value=0)
        config.validate()


def test_pattern_rule():
    """Test pattern rule application"""
    # Test simple addition
    rule = PatternRule([Operation.ADD], [2])
    assert rule.apply([1, 3], 1) == 5

    # Test composition
    rule = PatternRule([Operation.DOUBLE, Operation.ADD], [0, 3])
    assert rule.apply([1, 4], 1) == 11  # (4 * 2) + 3

    # Test rule composition
    rule1 = PatternRule([Operation.DOUBLE], [0])  # Double the number
    rule2 = PatternRule([Operation.ADD], [3])  # Add 3
    composed = PatternRule.compose([rule1, rule2])
    assert composed.apply([1, 4], 1) == 11  # (4 * 2) + 3


def test_sequence_dataset_deterministic():
    """Test that dataset generates same items with same seed"""
    config = NumberSequenceConfig(seed=42, size=10)
    dataset1 = NumberSequenceDataset(config)
    dataset2 = NumberSequenceDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_sequence_dataset_items():
    """Test basic properties of generated items"""
    config = NumberSequenceConfig(min_terms=4, max_terms=6, max_complexity=2, size=50, seed=42)
    dataset = NumberSequenceDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Verify sequence format
        question = item["question"]
        assert question.endswith(", ?")
        terms = [int(x) for x in question[:-3].split(", ")]
        assert len(terms) >= config.min_terms
        assert len(terms) <= config.max_terms


def test_sequence_dataset_iteration():
    """Test that iteration respects dataset size"""
    config = NumberSequenceConfig(size=5, seed=42)
    dataset = NumberSequenceDataset(config)

    items = list(dataset)
    assert len(items) == config.size

    # Test multiple iterations yield same items
    assert items == list(dataset)
