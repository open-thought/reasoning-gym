import random

import pytest

from reasoning_gym.algorithmic.ab import ABConfig, ABDataset, compute_steps, generate_program


def test_ab_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = ABConfig(length=0)
        config.validate()

    with pytest.raises(AssertionError):
        config = ABConfig(size=0)
        config.validate()


def test_ab_deterministic():
    """Test that dataset generates same items with same seed"""
    config = ABConfig(seed=42, size=10, length=5)
    dataset1 = ABDataset(config)
    dataset2 = ABDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_ab_program_generation():
    """Test program generation and computation"""
    rng = random.Random(42)
    program = generate_program(5, rng)

    # Test program format
    assert len(program) == 5
    assert all(token in ["A#", "#A", "B#", "#B"] for token in program)

    # Test computation
    steps, non_halting = compute_steps(program)
    assert isinstance(steps, list)
    assert isinstance(non_halting, bool)
    assert len(steps) > 0

    # Test each step follows valid transformation rules
    for step in steps:
        assert all(token in ["A#", "#A", "B#", "#B"] for token in step)


def test_ab_scoring():
    """Test scoring functionality"""
    config = ABConfig(seed=42, size=10, length=5)
    dataset = ABDataset(config)

    for item in dataset:
        # Test correct answer
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0

        # Test wrong answer
        wrong_answer = "A# B#" if item["answer"] != "A# B#" else "B# A#"
        assert dataset.score_answer(answer=wrong_answer, entry=item) == 0.01

        # Test None answer
        assert dataset.score_answer(answer=None, entry=item) == 0.0


def test_ab_iteration():
    """Test dataset iteration behavior"""
    config = ABConfig(size=5, seed=42)
    dataset = ABDataset(config)

    # Test length
    assert len(dataset) == config.size

    # Test iteration
    items = list(dataset)
    assert len(items) == config.size

    # Test multiple iterations yield same results
    items2 = list(dataset)
    assert items == items2


def test_ab_item_structure():
    """Test structure and content of generated items"""
    config = ABConfig(seed=42, size=10, length=5)
    dataset = ABDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Test question format
        assert "A::B is a system" in item["question"]
        assert "Return the final state" in item["question"]

        # Test answer format
        answer_tokens = item["answer"].split()
        assert all(token in ["A#", "#A", "B#", "#B"] for token in answer_tokens)
