import pytest
from reasoning_gym.arithmetic import FractionArithmeticDataset, FractionArithmeticConfig

def test_fraction_arithmetic_basic():
    """Test basic dataset properties and item structure."""
    config = FractionArithmeticConfig(size=10, seed=42)
    dataset = FractionArithmeticDataset(config)
    
    assert len(dataset) == 10
    
    item = dataset[0]
    assert "question" in item
    assert "answer" in item
    assert "metadata" in item
    
    # Verify format of answer (numerator/denominator)
    answer_parts = item["answer"].split("/")
    assert len(answer_parts) == 2
    assert int(answer_parts[1]) != 0

def test_fraction_arithmetic_seed_reproducibility():
    """Test that same seed produces identical items."""
    config1 = FractionArithmeticConfig(size=5, seed=123)
    config2 = FractionArithmeticConfig(size=5, seed=123)
    
    ds1 = FractionArithmeticDataset(config1)
    ds2 = FractionArithmeticDataset(config2)
    
    for i in range(5):
        assert ds1[i] == ds2[i]

def test_fraction_arithmetic_different_seeds():
    """Test that different seeds produce different items."""
    ds1 = FractionArithmeticDataset(FractionArithmeticConfig(size=5, seed=100))
    ds2 = FractionArithmeticDataset(FractionArithmeticConfig(size=5, seed=200))
    
    differences = sum(1 for i in range(5) if ds1[i] != ds2[i])
    assert differences > 0

def test_fraction_arithmetic_term_bounds():
    """Test that term bounds are respected."""
    config = FractionArithmeticConfig(
        min_terms=3,
        max_terms=3,
        min_denominator=4,
        max_denominator=6,
        size=5,
        seed=42
    )
    dataset = FractionArithmeticDataset(config)
    
    for i in range(len(dataset)):
        item = dataset[i]
        expr = item["metadata"]["expression"]
        ops_count = expr.count("+") + expr.count("-")
        assert ops_count == 2

def test_fraction_arithmetic_addition_only():
    """Test configuration with subtraction disabled."""
    config = FractionArithmeticConfig(
        allow_subtraction=False,
        size=5,
        seed=42
    )
    dataset = FractionArithmeticDataset(config)
    
    for i in range(len(dataset)):
        item = dataset[i]
        expr = item["metadata"]["expression"]
        assert "-" not in expr

def test_fraction_arithmetic_dataset_instantiation():
    """Test direct dataset instantiation with custom config."""
    config = FractionArithmeticConfig(size=2, seed=42)
    dataset = FractionArithmeticDataset(config)
    assert dataset[0] is not None