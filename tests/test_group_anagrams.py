"""Tests for Group Anagrams questions generation"""

import json

import pytest

from reasoning_gym.algorithmic.group_anagrams import GroupAnagramsConfig, GroupAnagramsDataset


def test_group_anagrams_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = GroupAnagramsConfig(anagram_groups=-1)  # Negative not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = GroupAnagramsConfig(anagram_groups=0)  # Zero not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = GroupAnagramsConfig(max_words_per_group=-1)  # Negative not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = GroupAnagramsConfig(max_words_per_group=0)  # Zero not allowed
        config.validate()


def test_group_anagrams_dataset_deterministic():
    """Test that dataset generates same items with same seed"""
    config = GroupAnagramsConfig(seed=42, size=10)
    dataset1 = GroupAnagramsDataset(config)
    dataset2 = GroupAnagramsDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_group_anagrams_dataset_items():
    """Test basic properties of generated items"""
    config = GroupAnagramsConfig(anagram_groups=5, max_words_per_group=3, size=10, seed=42)
    dataset = GroupAnagramsDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        # Check item structure
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata
        assert "words" in item["metadata"]
        assert "solution" in item["metadata"]

        words = item["metadata"]["words"]
        solution = item["metadata"]["solution"]

        # Verify list dimensions
        assert len(words) > 5
        assert len(solution) == 5
        assert all(len(group) <= 3 for group in solution)


def test_group_anagrams_dataset_iteration():
    """Test that iteration respects dataset size"""
    config = GroupAnagramsConfig(size=5, seed=42)
    dataset = GroupAnagramsDataset(config)

    items = list(dataset)
    assert len(items) == config.size

    # Test multiple iterations yield same items
    assert items == list(dataset)


def test_group_anagrams_answer():
    """Test the _group_anagrams method"""
    config = GroupAnagramsConfig(seed=42)
    dataset = GroupAnagramsDataset(config)

    # General use case
    words = ["eat", "tea", "tan", "ate", "nat", "bat"]
    correct = [["ate", "eat", "tea"], ["bat"], ["nat", "tan"]]
    assert json.dumps(dataset._group_anagrams(words)) == json.dumps(correct)

    # Single word
    words = ["a"]
    correct = [["a"]]
    assert json.dumps(dataset._group_anagrams(words)) == json.dumps(correct)

    # Empty list
    words = []
    correct = []
    assert json.dumps(dataset._group_anagrams(words)) == json.dumps(correct)


def test_group_anagrams_score_answer():
    """Test the score_answer method"""
    config = GroupAnagramsConfig(seed=42)
    dataset = GroupAnagramsDataset(config)

    # Verify the scoring function is permutation invariant
    answer = json.dumps([["bat"], ["nat", "tan"], ["ate", "eat", "tea"]])
    item = {"metadata": {"solution": [["ate", "eat", "tea"], ["bat"], ["nat", "tan"]]}}
    assert dataset.score_answer(answer, item) == 1

    # Verify the score is 0.01 when incorrect
    answer = json.dumps([["ate", "eat"], ["bat", "tea"], ["nat", "tan"]])
    item = {"metadata": {"solution": [["ate", "eat", "tea"], ["bat"], ["nat", "tan"]]}}
    assert dataset.score_answer(answer, item) == 0.01

    # Verify the score is 0 when answer is None
    answer = None
    item = {"metadata": {"solution": [["ate", "eat", "tea"], ["bat"], ["nat", "tan"]]}}
    assert dataset.score_answer(answer, item) == 0

    # Verify the score is 0 when answer is malformed JSON
    answer = '["ate", "eat", "tea"], ["bat"], ["nat", "tan"]'
    item = {"metadata": {"solution": [["ate", "eat", "tea"], ["bat"], ["nat", "tan"]]}}
    assert dataset.score_answer(answer, item) == 0
