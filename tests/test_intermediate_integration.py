"""Tests for intermediate integration task generation"""

import pytest
import sympy
from sympy.parsing.sympy_parser import parse_expr

from reasoning_gym.algebra.intermediate_integration import IntermediateIntegrationConfig, IntermediateIntegrationDataset


def test_intermediate_integration_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = IntermediateIntegrationConfig(problem_types=["invalid_problem_type"])
        config.validate()

    with pytest.raises(AssertionError):
        config = IntermediateIntegrationConfig(substitution_types=["invalid_substitution_type"])
        config.validate()

    with pytest.raises(AssertionError):
        config = IntermediateIntegrationConfig(by_parts_types=["invalid_by_parts_type"])
        config.validate()

    with pytest.raises(AssertionError):
        config = IntermediateIntegrationConfig(linear_lower_bound=2, linear_upper_bound=1)
        config.validate()

    with pytest.raises(AssertionError):
        config = IntermediateIntegrationConfig(linear_lower_bound=0)
        config.validate()

    with pytest.raises(AssertionError):
        config = IntermediateIntegrationConfig(min_linear_degree=5, max_linear_degree=1)
        config.validate()

    with pytest.raises(AssertionError):
        config = IntermediateIntegrationConfig(min_linear_degree=0)
        config.validate()

    with pytest.raises(AssertionError):
        config = IntermediateIntegrationConfig(outer_constant_min=5, outer_constant_max=1)
        config.validate()

    with pytest.raises(AssertionError):
        config = IntermediateIntegrationConfig(outer_constant_min=0)
        config.validate()

    with pytest.raises(AssertionError):
        config = IntermediateIntegrationConfig(min_poly_degree=5, max_poly_degree=1)
        config.validate()

    with pytest.raises(AssertionError):
        config = IntermediateIntegrationConfig(min_poly_degree=0)
        config.validate()

    with pytest.raises(AssertionError):
        config = IntermediateIntegrationConfig(symbols=("x", "y"))
        config.validate()

    with pytest.raises(AssertionError):
        config = IntermediateIntegrationConfig(operators=("+", "-", "*", "/"))
        config.validate()


def test_intermediate_integration_dataset_deterministic():
    """Test that dataset generates same items with same seed"""
    config = IntermediateIntegrationConfig(seed=42, size=10)
    dataset1 = IntermediateIntegrationDataset(config)
    dataset2 = IntermediateIntegrationDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_intermediate_integration_dataset_items():
    """Test that dataset items are valid"""
    config = IntermediateIntegrationConfig(seed=42, size=10)
    dataset = IntermediateIntegrationDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        assert "integrand" in item["metadata"]
        assert "problem_type" in item["metadata"]
        assert "variable" in item["metadata"]
        assert "type" in item["metadata"]

        # verify answer is mathematical expression
        answer = item["answer"]
        answer = answer.replace(" + C", "")
        assert isinstance(parse_expr(answer), sympy.Expr)


def test_solution_verification():
    """Test for solution verification of each answer"""
    config = IntermediateIntegrationConfig(seed=42, size=10)
    dataset = IntermediateIntegrationDataset(config)

    for item in dataset:
        integrand = parse_expr(item["metadata"]["integrand"])
        variable = sympy.Symbol(item["metadata"]["variable"])
        answer = parse_expr(item["answer"].replace(" + C", ""))

        # Verify that the derivative of the answer equals the integrand
        assert sympy.simplify(sympy.diff(answer, variable) - integrand) == 0
