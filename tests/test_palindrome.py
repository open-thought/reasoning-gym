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


def test_score_answer():
    """Test the scoring mechanism for palindrome answers.

    Expected behavior:
    - Correct answer (palindrome with only correct letters in the correct quantities) gives 1.0
    - An answer that is a palindrome, but not with the same letters as provided, gives 0.05
    - An answer that is a string, but not a palindrome gives 0.02
    - An empty string gives 0.01.
    - None gives 0.0.
    """
    config = PalindromeConfig(min_length=4, max_length=6, size=10, seed=42)
    dataset = PalindromeDataset(config)

    for item in dataset:
        correct_answer = item["answer"]

        # Correct answer should score 1.0
        assert dataset.score_answer(correct_answer, entry=item) == 1.0

        # Incorrect answer (palindrome, but not correct one) should score 0.05
        pal_letters = "racecar" if "racecar" != correct_answer else "aba"
        assert dataset.score_answer(pal_letters, entry=item) == 0.05

        # Incorrect answer (not palindrome) should score 0.02
        wrong_letters = "abcd" if "abcd" != correct_answer else "efgh"
        assert dataset.score_answer(wrong_letters, entry=item) == 0.02

        # Empty String input should score 0.01
        assert dataset.score_answer("", entry=item) == 0.01

        # Empty input should score 0.0
        assert dataset.score_answer(None, entry=item) == 0.0
