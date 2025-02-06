import pytest

from reasoning_gym.algebra.complex_arithmetic import ComplexArithmeticConfig, ComplexArithmeticDataset


def test_complex_arithmetic_basic():
    """Test basic functionality of complex arithmetic dataset."""
    config = ComplexArithmeticConfig(
        min_real=-5, max_real=5, min_imag=-5, max_imag=5, operations=("+", "-", "*", "/"), seed=42, size=10
    )
    dataset = ComplexArithmeticDataset(config)

    print(dataset)

    # Test dataset size
    assert len(dataset) == 10

    # Test a specific item
    item = dataset[0]
    assert "question" in item
    assert "answer" in item
    assert "metadata" in item

    # Add more detailed assertions
    assert isinstance(item["question"], str)
    assert isinstance(item["answer"], str)
    assert isinstance(item["metadata"], dict)

    # Check metadata structure
    assert "num1" in item["metadata"]
    assert "num2" in item["metadata"]
    assert "operation" in item["metadata"]
    assert "result" in item["metadata"]

    # Check data types in metadata
    assert isinstance(item["metadata"]["num1"], tuple)
    assert isinstance(item["metadata"]["num2"], tuple)
    assert len(item["metadata"]["num1"]) == 2  # Real and imaginary parts
    assert len(item["metadata"]["num2"]) == 2
    assert isinstance(item["metadata"]["operation"], str)
    assert isinstance(item["metadata"]["result"], tuple)

    # dump dataset into a text file
    with open("complex_arithmetic_dataset.txt", "w") as f:
        for item in dataset:
            f.write(str(item) + "\n")


def test_complex_arithmetic_scoring():
    """Test scoring function with various answer formats."""
    config = ComplexArithmeticConfig(seed=42)
    dataset = ComplexArithmeticDataset(config)

    # Create a test case with known answer
    metadata = {"result": (3.0, 2.0)}  # represents 3 + 2i

    # Test various correct answer formats
    assert dataset.score_answer("3 + 2i", metadata) == 1.0
    assert dataset.score_answer("3+2i", metadata) == 1.0
    assert dataset.score_answer("3.0 + 2.0i", metadata) == 1.0

    # Test incorrect answers
    assert dataset.score_answer("2 + 3i", metadata) == 0.0
    assert dataset.score_answer("3", metadata) == 0.0
    assert dataset.score_answer("inf + 2i", metadata) == 0.0
    assert dataset.score_answer("2i", metadata) == 0.0
    assert dataset.score_answer("invalid", metadata) == 0.0


def test_complex_arithmetic_division_by_zero():
    """Test that division by zero is handled properly."""
    config = ComplexArithmeticConfig(operations=("/",), seed=42)  # Only test division
    dataset = ComplexArithmeticDataset(config)

    # Check multiple items to ensure no division by zero
    for i in range(10):
        item = dataset[i]
        num2 = complex(*item["metadata"]["num2"])
        assert num2 != 0
