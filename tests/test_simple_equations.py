"""Tests for simple equation task generation"""

import pytest

from reasoning_gym.algebra.simple_equations import SimpleEquationsConfig, SimpleEquationsDataset


def test_simple_equations_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = SimpleEquationsConfig(min_terms=0)  # Too few terms
        config.validate()

    with pytest.raises(AssertionError):
        config = SimpleEquationsConfig(min_terms=5, max_terms=3)  # max < min terms
        config.validate()

    with pytest.raises(AssertionError):
        config = SimpleEquationsConfig(min_value=0)  # Too small value
        config.validate()

    with pytest.raises(AssertionError):
        config = SimpleEquationsConfig(min_value=100, max_value=50)  # max < min value
        config.validate()

    with pytest.raises(AssertionError):
        config = SimpleEquationsConfig(operators=())  # Empty operators
        config.validate()

    with pytest.raises(AssertionError):
        config = SimpleEquationsConfig(operators=("+", "^"))  # Invalid operator
        config.validate()


def test_simple_equations_dataset_deterministic():
    """Test that dataset generates same items with same seed"""
    config = SimpleEquationsConfig(seed=42, size=10)
    dataset1 = SimpleEquationsDataset(config)
    dataset2 = SimpleEquationsDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_simple_equations_dataset_items():
    """Test basic properties of generated items"""
    config = SimpleEquationsConfig(min_terms=2, max_terms=4, min_value=1, max_value=100, size=10, seed=42)
    dataset = SimpleEquationsDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        # Check item structure
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata
        assert "equation" in item["metadata"]
        assert "variable" in item["metadata"]

        # Verify answer is numeric (allowing negative numbers)
        answer = item["answer"]
        assert answer.replace("-", "").isdigit()

        # Verify equation format
        equation = item["metadata"]["equation"]
        assert "=" in equation
        assert item["metadata"]["variable"] in equation


def test_simple_equations_dataset_iteration():
    """Test that iteration respects dataset size"""
    config = SimpleEquationsConfig(size=5, seed=42)
    dataset = SimpleEquationsDataset(config)

    items = list(dataset)
    assert len(items) == config.size

    # Test multiple iterations yield same items
    assert items == list(dataset)


def test_simple_equations_solution_verification():
    """Test that generated equations have correct solutions"""
    config = SimpleEquationsConfig(
        min_terms=2,
        max_terms=3,
        min_value=1,
        max_value=10,  # Small values for predictable results
        operators=("+", "-"),  # Simple operators for easy verification
        size=10,
        seed=42,
    )
    dataset = SimpleEquationsDataset(config)

    for item in dataset:
        # Extract equation parts
        equation = item["metadata"]["equation"]
        variable = item["metadata"]["variable"]
        solution = int(item["answer"])

        # Verify solution by substitution
        equation_parts = equation.split("=")
        left_side = equation_parts[0].strip()
        right_side = int(equation_parts[1].strip())

        # Replace variable with solution
        evaluated = eval(left_side.replace(variable, str(solution)))
        assert evaluated == right_side


def test_simple_equations_operators():
    """Test equation generation with different operator combinations"""
    for operators in [
        ("+",),
        ("+", "-"),
        ("*",),
        ("+", "*"),
        ("+", "-", "*"),
    ]:
        config = SimpleEquationsConfig(operators=operators, size=5, seed=42)
        dataset = SimpleEquationsDataset(config)

        for item in dataset:
            equation = item["metadata"]["equation"]
            # Verify only allowed operators are used
            for op in "+-*":
                if op in equation:
                    assert op in operators, str(equation)
