import pytest
import sympy as sp

from reasoning_gym import create_dataset
from reasoning_gym.algebra.polynomial_multiplication import (
    PolynomialMultiplicationConfig,
    PolynomialMultiplicationDataset,
)


def test_polynomial_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        PolynomialMultiplicationConfig(min_terms=0).validate()

    with pytest.raises(AssertionError):
        PolynomialMultiplicationConfig(min_value=0).validate()

    with pytest.raises(AssertionError):
        PolynomialMultiplicationConfig(min_degree=0, max_degree=3).validate()

    with pytest.raises(AssertionError):
        PolynomialMultiplicationConfig(min_degree=4, max_degree=3).validate()

    with pytest.raises(AssertionError):
        PolynomialMultiplicationConfig(operators=("^",)).validate()

    with pytest.raises(AssertionError):
        PolynomialMultiplicationConfig(min_polynomials=5, max_polynomials=2).validate()


def test_polynomial_multiplication_dataset_basic():
    """Test dataset creation and length"""
    dataset_size = 50
    config = PolynomialMultiplicationConfig(
        min_terms=2,
        max_terms=3,
        min_value=1,
        max_value=5,
        min_degree=1,
        max_degree=2,
        min_polynomials=2,
        max_polynomials=3,
        single_variable=True,
        seed=42,
        size=dataset_size,
    )

    dataset = PolynomialMultiplicationDataset(config)

    assert len(dataset) == dataset_size


def test_polynomial_equations_dataset_items():
    """Test that generated items have correct structure"""
    ds = create_dataset(
        "polynomial_multiplication",
        min_terms=2,
        max_terms=3,
        min_value=1,
        max_value=5,
        min_degree=1,
        max_degree=2,
        min_polynomials=2,
        max_polynomials=5,
        single_variable=False,
        size=3,
        seed=100,
    )

    for item in ds:
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata
        assert isinstance(item["metadata"]["polynomial_expr"], str)
        assert isinstance(item["metadata"]["single_variable"], bool)

        # Check polynomial_expr existence
        poly_str = item["metadata"]["polynomial_expr"]
        # Ensure it can parse with sympy
        sp.sympify(poly_str)


def test_polynomial_equations_dataset_deterministic():
    """Test dataset reproducibility with fixed seed."""
    cfg = PolynomialMultiplicationConfig(seed=999, size=3)
    ds1 = PolynomialMultiplicationDataset(cfg)
    ds2 = PolynomialMultiplicationDataset(cfg)

    for i in range(len(ds1)):
        assert ds1[i] == ds2[i], "Polynomial datasets with same seed should match exactly."


def test_polynomial_solutions_evaluation():
    """Test that solution satisfy the polynomial multiplication."""
    ds = create_dataset(
        "polynomial_multiplication",
        min_terms=2,
        max_terms=4,
        min_value=1,
        max_value=10,
        min_degree=1,
        max_degree=3,
        min_polynomials=2,
        max_polynomials=5,
        single_variable=False,
        size=5,
        seed=42,
    )

    for item in ds:
        # Extract the polynomial expression
        poly_str = item["metadata"]["polynomial_expr"]
        # Get the polynomial product
        poly_expr = sp.expand(poly_str)

        # Verify that each solution satisfies the polynomial
        assert poly_expr == item["answer"]


def test_score_function():
    """Test that solution satisfy the polynomial multiplication."""
    ds = create_dataset(
        "polynomial_multiplication",
        min_terms=2,
        max_terms=4,
        min_value=1,
        max_value=10,
        min_degree=1,
        max_degree=3,
        min_polynomials=2,
        max_polynomials=5,
        single_variable=True,
        size=1,
        seed=42,
    )

    assert ds.score_answer(None, ds[0]) == 0.00
    assert ds.score_answer("6*x**4 + 9*x**3 - 6*x**2 - 39*x - 45", ds[0]) == 1
    assert ds.score_answer("Not a polynomial", ds[0]) == 0.01
    assert ds.score_answer("x**4", ds[0]) == 0.05


def test_multivariate_score_function():
    """Test that solution satisfy the polynomial multiplication."""
    ds = create_dataset(
        "polynomial_multiplication",
        min_terms=2,
        max_terms=4,
        min_value=1,
        max_value=10,
        min_degree=1,
        max_degree=3,
        min_polynomials=2,
        max_polynomials=5,
        single_variable=False,
        size=1,
        seed=42,
    )

    assert ds.score_answer(None, ds[0]) == 0.00
    assert ds.score_answer("-27*a**3*c - 27*a**3 + 144*a*c + 144*a", ds[0]) == 1
    assert ds.score_answer("Not a polynomial", ds[0]) == 0.01
    assert ds.score_answer("x**4", ds[0]) == 0.05
