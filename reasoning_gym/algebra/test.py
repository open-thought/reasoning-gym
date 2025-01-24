import pytest

from .simple_equations import simple_equations_dataset


def test_simple_equations_generation():
    dataset = simple_equations_dataset(seed=42, size=10)

    for item in dataset:
        # Check required keys exist
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Validate answer is a string of digits
        assert item["answer"].isdigit()

        # Validate equation format
        equation = item["metadata"]["equation"]
        variable = item["metadata"]["variable"]
        assert "=" in equation
        assert variable in equation

        # Validate question format
        question = item["question"]
        assert variable in question
        assert equation in question
        assert any(prompt in question for prompt in ["Find the value of", "Solve for", "Determine the value of"])


def test_simple_equations_config():
    # Test invalid config raises assertion
    with pytest.raises(AssertionError):
        dataset = simple_equations_dataset(min_terms=0)

    with pytest.raises(AssertionError):
        dataset = simple_equations_dataset(max_terms=1, min_terms=2)

    with pytest.raises(AssertionError):
        dataset = simple_equations_dataset(min_value=0)

    with pytest.raises(AssertionError):
        dataset = simple_equations_dataset(operators=())


def test_deterministic_generation():
    dataset1 = simple_equations_dataset(seed=42, size=5)
    dataset2 = simple_equations_dataset(seed=42, size=5)

    for i in range(5):
        assert dataset1[i]["question"] == dataset2[i]["question"]
        assert dataset1[i]["answer"] == dataset2[i]["answer"]
