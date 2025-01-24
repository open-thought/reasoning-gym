import pytest
from sympy import Symbol, sympify

from reasoning_gym import create_dataset
from reasoning_gym.algebra.polynomial_equations import PolynomialEquationsConfig, PolynomialEquationsDataset


def test_polynomial_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        PolynomialEquationsConfig(min_terms=0).validate()

    with pytest.raises(AssertionError):
        PolynomialEquationsConfig(min_value=0).validate()

    with pytest.raises(AssertionError):
        PolynomialEquationsConfig(min_degree=0, max_degree=3).validate()

    with pytest.raises(AssertionError):
        PolynomialEquationsConfig(min_degree=4, max_degree=3).validate()

    with pytest.raises(AssertionError):
        PolynomialEquationsConfig(operators=("^",)).validate()


def test_polynomial_equations_dataset_basic():
    """Test dataset creation and length"""
    dataset_size = 50
    config = PolynomialEquationsConfig(
        min_terms=2,
        max_terms=3,
        min_value=1,
        max_value=5,
        min_degree=1,
        max_degree=2,
        seed=42,
        size=dataset_size,
    )

    dataset = PolynomialEquationsDataset(config)

    assert len(dataset) == dataset_size


def test_polynomial_equations_dataset_items():
    """Test that generated items have correct structure"""
    ds = create_dataset(
        "polynomial_equations",
        min_terms=2,
        max_terms=3,
        min_value=1,
        max_value=5,
        min_degree=1,
        max_degree=2,
        size=3,
        seed=100,
    )

    for item in ds:
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata
        assert isinstance(item["metadata"]["polynomial_expr"], str)
        assert isinstance(item["metadata"]["variable"], str)
        assert isinstance(item["metadata"]["degree"], int)
        assert isinstance(item["metadata"]["real_solutions"], list)

        # Check polynomial_expr existence
        poly_str = item["metadata"]["polynomial_expr"]
        # Ensure it can parse with sympy
        sympify(poly_str)


def test_polynomial_equations_dataset_deterministic():
    """Test dataset reproducibility with fixed seed."""
    cfg = PolynomialEquationsConfig(seed=999, size=3)
    ds1 = PolynomialEquationsDataset(cfg)
    ds2 = PolynomialEquationsDataset(cfg)

    for i in range(len(ds1)):
        assert ds1[i] == ds2[i], "Polynomial datasets with same seed should match exactly."


def test_polynomial_solutions_evaluation():
    """Test that real_solutions satisfy the polynomial equation."""
    ds = create_dataset(
        "polynomial_equations",
        min_terms=2,
        max_terms=4,
        min_value=1,
        max_value=10,
        min_degree=1,
        max_degree=3,
        size=5,
        seed=42,
    )

    for item in ds:
        # Extract the polynomial expression and solutions
        poly_str = item["metadata"]["polynomial_expr"]
        real_solutions = item["metadata"]["real_solutions"]
        x = Symbol(item["metadata"]["variable"])
        # Parse the polynomial expression
        poly_expr = sympify(poly_str)

        # Verify that each solution satisfies the polynomial
        for solution in real_solutions:
            # Evaluate the expression with the solution substituted
            evaluated_value = poly_expr.subs(x, solution)

            # Ensure the evaluated value is close to zero (numerical stability threshold)
            assert abs(evaluated_value) < 1e-6, (
                f"Solution {solution} does not satisfy the polynomial {poly_str}. "
                f"Evaluated value: {evaluated_value}"
            )
