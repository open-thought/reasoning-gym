import pytest

from reasoning_gym.algorithmic.palindrome_generation import PalindromeConfig, PalindromeDataset


def test_palindrome_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = PalindromeConfig(min_length=0)  # Too short
        config.validate()

    with pytest.raises(AssertionError):
        config = PalindromeConfig(min_length=5, max_length=3)  # Invalid range
        config.validate()


def test_palindrome_deterministic():
    """Test that dataset generates same items with same seed"""
    config = PalindromeConfig(seed=42, size=10)
    dataset1 = PalindromeDataset(config)
    dataset2 = PalindromeDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_palindrome_items():
    """Test basic properties of generated items"""
    config = PalindromeConfig(min_length=3, max_length=7, size=10, seed=42)
    dataset = PalindromeDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata contains required fields
        assert "letters" in item["metadata"]
        assert "generated_palindrome" in item["metadata"]

        # Verify answer is a palindrome
        palindrome = item["answer"]
        assert palindrome == palindrome[::-1], f"{palindrome} is not a palindrome"


def test_palindrome_randomization():
    """Test letter randomization in the question"""
    config = PalindromeConfig(min_length=4, max_length=4, size=10, seed=42)
    dataset = PalindromeDataset(config)

    for item in dataset:
        letters = item["metadata"]["letters"]
        palindrome = item["metadata"]["generated_palindrome"]

        # Ensure the same letters are present but in different order
        assert sorted(letters) == sorted(palindrome)
