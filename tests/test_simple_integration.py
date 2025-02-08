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
        assert "expected_answer_expression" in item["metadata"]

        # Verify answer is a mathematical expression
        answer = item["answer"]
        answer = answer.replace(" + C", "")
        assert isinstance(parse_expr(answer), sympy.Expr)


def test_verify_answer():
    config = SimpleIntegrationConfig(seed=42)
    dataset = SimpleIntegrationDataset(config)
    for i in range(len(dataset)):
        item = dataset[i]
        score = dataset.score_answer(item["answer"], item)
        assert score == 1.0


def test_score_answer_cases():
    """Test various answer scoring scenarios"""
    config = SimpleIntegrationConfig(seed=42)
    dataset = SimpleIntegrationDataset(config)
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
        dummy_entry = {"metadata": metadata}
        score = dataset.score_answer(answer=answer, entry=dummy_entry)
        assert score == expected, f"Failed case: {answer} | Expected {expected}, got {score}"
