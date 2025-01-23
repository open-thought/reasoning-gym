import pytest
from reasoning_gym.arithmetic import ChainSum, ChainSumConfig


def test_chain_sum_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = ChainSumConfig(min_terms=0)
        config.validate()
    
    with pytest.raises(AssertionError):
        config = ChainSumConfig(min_terms=3, max_terms=2)
        config.validate()


def test_chain_sum_deterministic():
    """Test that dataset generates same items with same seed"""
    config = ChainSumConfig(seed=42, size=10)
    dataset1 = ChainSum(config)
    dataset2 = ChainSum(config)
    
    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_chain_sum_items():
    """Test basic properties of generated items"""
    config = ChainSumConfig(
        min_terms=2,
        max_terms=4,
        min_digits=1,
        max_digits=2,
        size=100,
        seed=42
    )
    dataset = ChainSum(config)
    
    for i in range(len(dataset)):
        item = dataset[i]
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item
        
        # Verify only + and - are used
        expression = item["metadata"]["expression"]
        assert all(op in ["+", "-", " "] or op.isdigit() for op in expression)
        
        # Verify the answer matches the expression
        answer = eval(expression)  # Safe here as we control the expression
        assert str(answer) == item["answer"]
