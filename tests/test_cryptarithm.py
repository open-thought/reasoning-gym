import pytest

from reasoning_gym import create_dataset
from reasoning_gym.algorithmic.cryptarithm import CryptarithmConfig, CryptarithmDataset


def test_cryptarithm_generation():
    dataset = create_dataset("cryptarithm", seed=42, size=10)
    assert isinstance(dataset, CryptarithmDataset)
    unique_number = set()
    for item in dataset:
        # Check required keys exist
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Validate question format
        question = item["question"]
        assert "Solve this cryptarithm:" in question
        assert "Each letter stands for a unique digit (0-9)" in question

        # Validate metadata structure
        metadata = item["metadata"]
        assert "letters" in metadata
        assert "letter_to_digit" in metadata
        assert "words_letters" in metadata
        assert "result_letters" in metadata
        assert "word_values" in metadata
        assert "sum_number" in metadata

        # Validate letter to digit mapping
        letter_to_digit = metadata["letter_to_digit"]
        used_digits = set(letter_to_digit.values())
        assert len(used_digits) == len(letter_to_digit), "Each letter should map to a unique digit"
        assert all(0 <= digit <= 9 for digit in used_digits), "All digits should be between 0 and 9"

        # Validate the arithmetic
        word_values = metadata["word_values"]
        result_value = metadata["sum_number"]
        assert sum(word_values) == result_value, "Sum of word values should equal result value"
        unique_number.add(result_value)

    assert len(unique_number) == len(dataset)


def test_cryptarithm_config():
    # Test invalid configs raise assertions
    with pytest.raises(AssertionError):
        dataset = create_dataset("cryptarithm", min_words=1)  # min_words must be >= 2

    with pytest.raises(AssertionError):
        dataset = create_dataset("cryptarithm", min_words=4, max_words=3)  # min must be <= max

    with pytest.raises(AssertionError):
        dataset = create_dataset("cryptarithm", size=0)  # size must be positive


def test_leading_zero_constraint():
    # Test with leading zeros not allowed
    dataset = create_dataset("cryptarithm", seed=42, size=5, allow_leading_zero=False, max_words=10, min_words=5)

    for item in dataset:
        # print(item['question'])
        metadata = item["metadata"]
        letter_to_digit = metadata["letter_to_digit"]
        words_letters = metadata["words_letters"]
        result_letters = metadata["result_letters"]

        # Check leading letters of all words and result
        leading_letters = [word[0] for word in words_letters] + [result_letters[0]]
        for letter in leading_letters:
            assert letter_to_digit[letter] != 0, "Leading letters cannot be zero when allow_leading_zero=False"


def test_deterministic_generation():
    dataset1 = create_dataset("cryptarithm", seed=42, size=5)
    dataset2 = create_dataset("cryptarithm", seed=42, size=5)

    for i in range(5):
        assert dataset1[i]["question"] == dataset2[i]["question"]
        assert dataset1[i]["answer"] == dataset2[i]["answer"]
        assert dataset1[i]["metadata"] == dataset2[i]["metadata"]


def test_word_length_constraints():
    dataset = create_dataset("cryptarithm", seed=42, size=10)

    for item in dataset:
        metadata = item["metadata"]
        words_letters = metadata["words_letters"]

        # Check each word is between 3-5 letters as specified in the code
        for word in words_letters:
            assert 3 <= len(word) <= 5, "Each word should be between 3 and 5 letters long"


def test_max_letters_constraint():
    dataset = create_dataset("cryptarithm", seed=42, size=10)

    for item in dataset:
        metadata = item["metadata"]
        letter_to_digit = metadata["letter_to_digit"]

        # Check total unique letters doesn't exceed 10 (digits 0-9)
        assert len(letter_to_digit) <= 10, "Total unique letters should not exceed 10"
