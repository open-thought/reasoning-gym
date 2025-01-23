"""Tests for propositional logic task generation"""

import pytest

from reasoning_gym.logic.propositional_logic import (
    Expression,
    Operator,
    PropositionalLogicConfig,
    PropositionalLogicDataset,
)


def test_propositional_logic_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = PropositionalLogicConfig(min_vars=0)
        config.validate()

    with pytest.raises(AssertionError):
        config = PropositionalLogicConfig(min_vars=4, max_vars=3)
        config.validate()

    with pytest.raises(AssertionError):
        config = PropositionalLogicConfig(min_statements=0)
        config.validate()


def test_expression_evaluation():
    """Test logical expression evaluation"""
    # Test simple variable
    expr = Expression(None, "P")
    assert expr.evaluate({"P": True}) is True
    assert expr.evaluate({"P": False}) is False

    # Test NOT
    expr = Expression(Operator.NOT, Expression(None, "P"))
    assert expr.evaluate({"P": True}) is False
    assert expr.evaluate({"P": False}) is True

    # Test AND
    expr = Expression(Operator.AND, Expression(None, "P"), Expression(None, "Q"))
    assert expr.evaluate({"P": True, "Q": True}) is True
    assert expr.evaluate({"P": True, "Q": False}) is False

    # Test IMPLIES
    expr = Expression(Operator.IMPLIES, Expression(None, "P"), Expression(None, "Q"))
    assert expr.evaluate({"P": True, "Q": False}) is False
    assert expr.evaluate({"P": True, "Q": True}) is True
    assert expr.evaluate({"P": False, "Q": False}) is True


def test_propositional_logic_dataset_deterministic():
    """Test that dataset generates same items with same seed"""
    config = PropositionalLogicConfig(seed=42, size=10)
    dataset1 = PropositionalLogicDataset(config)
    dataset2 = PropositionalLogicDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_propositional_logic_dataset_items():
    """Test basic properties of generated items"""
    config = PropositionalLogicConfig(
        min_vars=2, max_vars=3, min_statements=2, max_statements=3, max_complexity=2, size=10, seed=42
    )
    dataset = PropositionalLogicDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item
        assert isinstance(item["metadata"]["premises"], list)
        assert isinstance(item["metadata"]["variables"], list)
        assert isinstance(item["metadata"]["complexity"], int)


def test_propositional_logic_dataset_iteration():
    """Test that iteration respects dataset size"""
    config = PropositionalLogicConfig(size=5, seed=42)
    dataset = PropositionalLogicDataset(config)

    items = list(dataset)
    assert len(items) == config.size

    # Test multiple iterations yield same items
    assert items == list(dataset)
