"""Tests for Group Anagrams questions generation"""

import json

import pytest

from reasoning_gym.algorithmic.group_anagrams import GroupAnagramsConfig, GroupAnagramsCurriculum, GroupAnagramsDataset


def test_group_anagrams_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = GroupAnagramsConfig(min_anagram_groups=-1)  # Negative not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = GroupAnagramsConfig(min_anagram_groups=0)  # Zero not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = GroupAnagramsConfig(min_anagram_groups=5, max_anagram_groups=4)  # Min > Max not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = GroupAnagramsConfig(min_words_per_group=-1)  # Negative not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = GroupAnagramsConfig(min_words_per_group=0)  # Zero not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = GroupAnagramsConfig(min_words_per_group=5, max_words_per_group=4)  # Min > Max not allowed
        config.validate()  # Min > Max not allowed


def test_group_anagrams_dataset_deterministic():
    """Test that dataset generates same items with same seed"""
    config = GroupAnagramsConfig(seed=42, size=10)
    dataset1 = GroupAnagramsDataset(config)
    dataset2 = GroupAnagramsDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_group_anagrams_dataset_items():
    """Test basic properties of generated items"""
    config = GroupAnagramsConfig(max_anagram_groups=5, max_words_per_group=3, size=10, seed=42)
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
        assert len(words) >= len(solution)
        assert len(solution) <= 5
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


def test_group_anagrams_curriculum():
    curriculum = GroupAnagramsCurriculum()

    base_value = {"size": 150, "seed": 1}

    base_cfg: GroupAnagramsConfig = curriculum.generate_configuration(base_value)
    assert base_cfg.seed == 1
    assert base_cfg.size == 150
    assert base_cfg.min_anagram_groups == 10 and base_cfg.max_anagram_groups == 10
    assert base_cfg.min_words_per_group == 2 and base_cfg.max_words_per_group == 2

    # test incrementing attribute levels
    curriculum.increment_attr_level("anagram_groups")
    curriculum.increment_attr_level("words_per_group")
    increased_cfg = curriculum.generate_configuration(base_value)
    assert increased_cfg.min_anagram_groups == 10 and increased_cfg.max_anagram_groups == 100
    assert increased_cfg.min_words_per_group == 2 and increased_cfg.max_words_per_group == 5

    # test decrementing attribute level partially
    curriculum.decrement_attr_level("anagram_groups")
    partially_decreased_cfg = curriculum.generate_configuration(base_value)
    assert partially_decreased_cfg.min_anagram_groups == 10 and partially_decreased_cfg.max_anagram_groups == 10
    assert partially_decreased_cfg.min_words_per_group == 2 and partially_decreased_cfg.max_words_per_group == 5
