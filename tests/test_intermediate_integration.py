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


def test_verify_answer():
    config = IntermediateIntegrationConfig(seed=42)
    dataset = IntermediateIntegrationDataset(config)
    for i in range(len(dataset)):
        item = dataset[i]
        score = dataset.score_answer(item["answer"], item["metadata"])
        assert score == 1.0


def test_score_answer_cases():
    """Test various answer scoring scenarios"""
    config = IntermediateIntegrationConfig(seed=42)
    dataset = IntermediateIntegrationDataset(config)
    x = sympy.Symbol("x")
    X = sympy.Symbol("X")

    # Test cases: (answer, metadata, expected_score)
    test_cases = [
        # Correct answers
        ("x**2 + C", {"variable": "x", "integrand": "2*x"}, 1.0),
        ("X**3 - 5*X + C", {"variable": "X", "integrand": "3*X**2 - 5"}, 1.0),
        ("sin(x) + C", {"variable": "x", "integrand": "cos(x)"}, 1.0),
        # Correct without explicit constant
        ("x**2", {"variable": "x", "integrand": "2*x"}, 1.0),
        ("log(x)", {"variable": "x", "integrand": "1/x"}, 1.0),
        # Incorrect but properly formatted
        ("x**3 + C", {"variable": "x", "integrand": "2*x"}, 0.05),
        ("cos(X)", {"variable": "X", "integrand": "sin(X)"}, 0.05),
        # Malformed expressions
        ("x**2 +", {"variable": "x", "integrand": "2*x"}, 0.01),
        ("sin(x", {"variable": "x", "integrand": "cos(x)"}, 0.01),
        # Empty answer
        ("", {"variable": "x", "integrand": "2*x"}, 0.01),
        # Case sensitivity
        ("x**2 + C", {"variable": "X", "integrand": "2*X"}, 0.05),
        ("X**2 + C", {"variable": "x", "integrand": "2*x"}, 0.05),
        # Alternative constant notation
        ("x**2 + K", {"variable": "x", "integrand": "2*x"}, 1.0),
        ("sin(x) + D", {"variable": "x", "integrand": "cos(x)"}, 1.0),
        # Simplification required
        ("x**2 + C + 5 - 5", {"variable": "x", "integrand": "2*x"}, 1.0),
        ("(x**3)/3 - 2*x + C", {"variable": "x", "integrand": "x**2 - 2"}, 1.0),
    ]

    for answer, metadata, expected in test_cases:
        score = dataset.score_answer(answer, metadata)
        assert score == expected, f"Failed case: {answer} | Expected {expected}, got {score}"
