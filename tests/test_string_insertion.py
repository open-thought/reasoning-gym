"""Tests for String Insertion questions generation"""

import pytest

from reasoning_gym.algorithmic.string_insertion import StringInsertionConfig, StringInsertionDataset


def test_string_insertion_config_validation():
    """Test that invalid configs raise appropriate errors"""

    for field in ["min_string_length", "max_string_length"]:
        for i in range(-1, 5):
            with pytest.raises(AssertionError):
                config = StringInsertionConfig(**{field: i})  # [-1, 4] is invalid
                config.validate()


def test_string_insertion_dataset_deterministic():
    """Test that dataset generates same items with same seed"""
    config = StringInsertionConfig(seed=42, size=10)
    dataset1 = StringInsertionDataset(config)
    dataset2 = StringInsertionDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_string_insertion_dataset_items():
    """Test basic properties of generated items"""
    config = StringInsertionConfig(min_string_length=5, max_string_length=30, size=10, seed=42)
    dataset = StringInsertionDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        # Check item structure
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata
        assert "string" in item["metadata"]
        assert "solution" in item["metadata"]

        string = item["metadata"]["string"]
        solution = item["metadata"]["solution"]

        # Verify string dimensions
        assert 5 <= len(string) <= 30
        assert len(string) <= len(solution)


def test_string_insertion_dataset_iteration():
    """Test that iteration respects dataset size"""
    config = StringInsertionConfig(size=5, seed=42)
    dataset = StringInsertionDataset(config)

    items = list(dataset)
    assert len(items) == config.size

    # Test multiple iterations yield same items
    assert items == list(dataset)


def test_string_insertion_answer():
    """Test the _get_rotated method"""
    config = StringInsertionConfig(seed=42)
    dataset = StringInsertionDataset(config)

    # No pattern match
    assert dataset._get_answer("AAAAAAA") == "AAAAAAA"
    assert dataset._get_answer("ADBEEBEA") == "ADBEEBEA"
    assert dataset._get_answer("ADEACA") == "ADEACA"

    # Insert A after ABCD
    assert dataset._get_answer("ABCDE") == "ABCDAE"

    # Insert B after BCDE
    assert dataset._get_answer("AEBCDEC") == "AEBCDEBC"

    # Insert C after CDEA
    assert dataset._get_answer("BBACDEAC") == "BBACDEACC"

    # Insert D after DEAB
    assert dataset._get_answer("BAAABDEAB") == "BAAABDEABD"

    # Insert E after EABC
    assert dataset._get_answer("EABCBCBC") == "EABCEBCBC"

    # Multiple insertions
    assert dataset._get_answer("AABCDEEEEEEEBCDEAAAAA") == "AABCDAEEEEEEEBCDEBAAAAA"

    # No reuse of newly inserted characters
    assert dataset._get_answer("ABCDBCD") == "ABCDABCD"

    # Test score_answer with correct answer
    answer = "AABCDAEEEEEEEBCDEBAAAAA"
    entry = {"answer": "AABCDAEEEEEEEBCDEBAAAAA"}
    assert dataset.score_answer(answer, entry) == 1.0

    # Test score_answer with correct answer as python list of characters (partial correct)
    answer = "['A', 'A', 'B', 'C', 'D', 'A', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'B', 'C', 'D', 'E', 'B', 'A', 'A', 'A', 'A', 'A']"
    entry = {"answer": "AABCDAEEEEEEEBCDEBAAAAA"}
    assert dataset.score_answer(answer, entry) == 0.5
