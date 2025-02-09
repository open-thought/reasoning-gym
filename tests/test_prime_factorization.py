"""Tests for prime factorization task generation"""

import pytest

from reasoning_gym.arithmetic.prime_factorization import PrimeFactorizationConfig, PrimeFactorizationDataset


def test_prime_factorization_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = PrimeFactorizationConfig(min_value=1)  # Too small
        config.validate()

    with pytest.raises(AssertionError):
        config = PrimeFactorizationConfig(min_value=100, max_value=50)  # max < min
        config.validate()


def test_prime_factorization_dataset_deterministic():
    """Test that dataset generates same items with same seed"""
    config = PrimeFactorizationConfig(seed=42, size=10)
    dataset1 = PrimeFactorizationDataset(config)
    dataset2 = PrimeFactorizationDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_prime_factorization_dataset_items():
    """Test basic properties of generated items"""
    config = PrimeFactorizationConfig(min_value=2, max_value=100, size=10, seed=42)
    dataset = PrimeFactorizationDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        # Check item structure
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata
        assert "number" in item["metadata"]
        assert "factors" in item["metadata"]

        # Verify value range
        number = item["metadata"]["number"]
        assert config.min_value <= number <= config.max_value

        # Verify factorization is correct
        factors = item["metadata"]["factors"]
        product = 1
        for factor in factors:
            product *= factor
        assert product == number

        # Verify factors are prime
        for factor in factors:
            assert is_prime(factor), f"{factor} is not prime"

        # Verify answer format
        assert item["answer"] == " × ".join(map(str, factors))


def test_prime_factorization_dataset_iteration():
    """Test that iteration respects dataset size"""
    config = PrimeFactorizationConfig(size=5, seed=42)
    dataset = PrimeFactorizationDataset(config)

    items = list(dataset)
    assert len(items) == config.size

    # Test multiple iterations yield same items
    assert items == list(dataset)


def test_prime_factorization_known_values():
    """Test factorization of known values"""
    config = PrimeFactorizationConfig(min_value=12, max_value=12, size=1, seed=42)  # Force specific number
    dataset = PrimeFactorizationDataset(config)
    item = dataset[0]

    assert item["metadata"]["number"] == 12
    assert item["metadata"]["factors"] == [2, 2, 3]
    assert item["answer"] == "2 × 2 × 3"


def test_prime_factorization_score_answer():
    """Test scoring of answers"""
    config = PrimeFactorizationConfig(min_value=12, max_value=12, size=1, seed=42)  # Force specific number
    dataset = PrimeFactorizationDataset(config)
    item = dataset[0]

    # Perfectly ordered answer
    answer = "2 × 2 × 3"
    assert dataset.score_answer(answer, item) == 1.0

    # No white spaces answer (still correct)
    answer = "2×2×3"
    assert dataset.score_answer(answer, item) == 1.0

    # Shuffled factors (still correct)
    answer = "2 × 3 × 2"
    assert dataset.score_answer(answer, item) == 1.0

    # Partially correct answer (not all numbers are fully factorized)
    answer = "2 × 6"
    assert dataset.score_answer(answer, item) == 0.5

    # Incorrect answer
    answer = "2 × 5"
    assert dataset.score_answer(answer, item) == 0.01

    # Answer is none
    answer = None
    assert dataset.score_answer(answer, item) == 0.0


def is_prime(n: int) -> bool:
    """Helper function to check if a number is prime"""
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
