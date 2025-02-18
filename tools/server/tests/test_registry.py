"""Tests for experiment registry."""

import pytest
from reasoning_gym.composite import CompositeConfig
from reasoning_gym.arithmetic.chain_sum import ChainSumConfig
from ..registry import ExperimentRegistry


def test_singleton():
    """Test that ExperimentRegistry is a singleton."""
    registry1 = ExperimentRegistry()
    registry2 = ExperimentRegistry()
    assert registry1 is registry2


def test_experiment_management():
    """Test basic experiment management operations."""
    registry = ExperimentRegistry()
    
    # Clear any existing experiments
    for name in registry.list_experiments():
        registry.remove_experiment(name)
    
    # Test registration with chain_sum dataset
    config = CompositeConfig(
        size=10, 
        seed=42,
        datasets={
            "chain_sum": {
                "config": ChainSumConfig(size=10, seed=42),
                "weight": 1.0
            }
        }
    )
    registry.register_experiment("test_exp", config)
    
    # Test listing
    assert "test_exp" in registry.list_experiments()
    
    # Test retrieval
    exp = registry.get_experiment("test_exp")
    assert exp is not None
    
    # Test removal
    assert registry.remove_experiment("test_exp")
    assert "test_exp" not in registry.list_experiments()
    assert not registry.remove_experiment("nonexistent")
