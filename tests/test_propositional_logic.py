"""Tests for propositional logic task generation"""

import pytest

from reasoning_gym.logic.propositional_logic import (
    Expression,
    Operator,
    PropositionalLogicConfig,
    PropositionalLogicCurriculum,
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


def test_propositional_logic_dataset_score_answer_correct():
    dataset = PropositionalLogicDataset(PropositionalLogicConfig(size=50, seed=101))
    for i, item in enumerate(dataset):
        score = dataset.score_answer(item["metadata"]["example_answer"], item)
        assert score == 1.0


def test_propositional_logic_dataset_score_answer_incorrect():
    dataset = PropositionalLogicDataset(PropositionalLogicConfig(size=100, seed=101))
    for i, item in enumerate(dataset):
        score = dataset.score_answer("Wrong", item)
        assert score == 0.0


def test_propositional_logic_curriculum():
    curriculum = PropositionalLogicCurriculum()

    base_value = {"size": 150, "seed": 1}

    base_cfg: PropositionalLogicConfig = curriculum.generate_configuration(base_value)
    assert base_cfg.seed == 1
    assert base_cfg.size == 150
    assert base_cfg.min_vars == 2 and base_cfg.max_vars == 2
    assert base_cfg.min_statements == 2 and base_cfg.max_statements == 2
    assert base_cfg.min_complexity == 1 and base_cfg.max_complexity == 1

    # test incrementing attribute levels
    curriculum.increment_attr_level("vars")
    curriculum.increment_attr_level("statements")
    curriculum.increment_attr_level("complexity")
    increased_cfg = curriculum.generate_configuration(base_value)
    assert increased_cfg.min_vars == 2 and increased_cfg.max_vars == 4
    assert increased_cfg.min_statements == 2 and increased_cfg.max_statements == 4
    assert increased_cfg.min_complexity == 1 and increased_cfg.max_complexity == 2

    # test decrementing attribute level for vars again
    curriculum.decrement_attr_level("vars")
    partially_decreased_cfg = curriculum.generate_configuration(base_value)
    assert partially_decreased_cfg.min_vars == 2 and partially_decreased_cfg.max_vars == 2
    assert partially_decreased_cfg.min_statements == 2 and partially_decreased_cfg.max_statements == 4
    assert partially_decreased_cfg.min_complexity == 1 and partially_decreased_cfg.max_complexity == 2
