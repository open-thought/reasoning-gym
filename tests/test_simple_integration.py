import random
from fractions import Fraction

import pytest
import sympy
from sympy.parsing.sympy_parser import parse_expr

from reasoning_gym.algebra.simple_integration import SimpleIntegrationConfig, SimpleIntegrationDataset


def test_simple_integration_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = SimpleIntegrationConfig(min_bounds=0)
        config.validate()

    with pytest.raises(AssertionError):
        config = SimpleIntegrationConfig(max_bounds=5, min_bounds=10)
        config.validate()

    with pytest.raises(AssertionError):
        config = SimpleIntegrationConfig(min_terms=-1)
        config.validate()

    with pytest.raises(AssertionError):
        config = SimpleIntegrationConfig(max_terms=2, min_terms=5)
        config.validate()

    with pytest.raises(AssertionError):
        config = SimpleIntegrationConfig(min_degree=-11)
        config.validate()

    with pytest.raises(AssertionError):
        config = SimpleIntegrationConfig(max_degree=3, min_degree=5)
        config.validate()

    with pytest.raises(AssertionError):
        config = SimpleIntegrationConfig(operators=("+", "-", "*"))
        config.validate()


def test_simple_integration_dataset_deterministic():
    """Test that dataset generates same items with same seed"""
    config = SimpleIntegrationConfig(seed=42, size=10)
    dataset1 = SimpleIntegrationDataset(config)
    dataset2 = SimpleIntegrationDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_simple_integration_dataset_items():
    """Test that dataset items are valid"""
    config = SimpleIntegrationConfig(seed=42, size=10)
    dataset = SimpleIntegrationDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        assert "integrand" in item["metadata"]
        assert "variable" in item["metadata"]
        assert "antiderivative" in item["metadata"]

        # Verify answer is a mathematical expression
        answer = item["answer"]
        answer = answer.replace(" + C", "")
        assert isinstance(parse_expr(answer), sympy.Expr)


def test_simple_integration_solution_verification():
    """Test for solution verification of each answer"""
    config = SimpleIntegrationConfig(seed=42, size=10)
    dataset = SimpleIntegrationDataset(config)

    for item in dataset:
        integrand = parse_expr(item["metadata"]["integrand"])
        variable = sympy.Symbol(item["metadata"]["variable"])
        answer = parse_expr(item["answer"].replace(" + C", ""))

        # Verify that the derivative of the answer equals the integrand
        assert sympy.simplify(sympy.diff(answer, variable) - integrand) == 0
