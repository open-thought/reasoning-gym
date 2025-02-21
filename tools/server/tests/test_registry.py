"""Tests for experiment registry."""

import pytest

from reasoning_gym.arithmetic.chain_sum import ChainSumConfig
from reasoning_gym.coaching.registry import ExperimentRegistry
from reasoning_gym.composite import CompositeConfig, CompositeDataset, DatasetSpec


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
    chain_sum_spec = DatasetSpec(name="chain_sum", weight=1.0, config=vars(ChainSumConfig(size=10, seed=42)))

    config = CompositeConfig(size=10, seed=42, datasets=[chain_sum_spec])
    registry.register_experiment("test_exp", config)

    # Test listing
    assert "test_exp" in registry.list_experiments()

    # Test retrieval
    exp = registry.get_experiment("test_exp")
    assert exp is not None
    assert exp.name == "test_exp"
    assert isinstance(exp.dataset, CompositeDataset)
    assert exp.config == config

    # Test removal
    assert registry.remove_experiment("test_exp")
    assert "test_exp" not in registry.list_experiments()
    assert not registry.remove_experiment("nonexistent")
