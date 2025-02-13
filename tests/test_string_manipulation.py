"""Tests for String Manipulation questions generation"""

import pytest

from reasoning_gym.algorithmic.string_manipulation import StringManipulationConfig, StringManipulationDataset


def test_string_manipulation_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = StringManipulationConfig(min_string_length=4)  # Minimum string length should be at least 5
        config.validate()

    with pytest.raises(AssertionError):
        config = StringManipulationConfig(min_string_length=10, max_string_length=7)  # Max must be greater than min
        config.validate()

    with pytest.raises(AssertionError):
        config = StringManipulationConfig(min_num_rules=2)  # Min number of rules should be at least 3
        config.validate()

    with pytest.raises(AssertionError):
        config = StringManipulationConfig(min_num_rules=5, max_num_rules=3)  # Max must be greater than min
        config.validate()


def test_string_manipulation_dataset_deterministic():
    """Test that dataset generates same items with same seed"""
    config = StringManipulationConfig(seed=42, size=10)
    dataset1 = StringManipulationDataset(config)
    dataset2 = StringManipulationDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_string_manipulation_dataset_items():
    """Test basic properties of generated items"""
    config = StringManipulationConfig(
        min_string_length=7, max_string_length=25, min_num_rules=5, max_num_rules=12, size=10, seed=42
    )
    dataset = StringManipulationDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        # Check item structure
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata
        assert "string" in item["metadata"]
        assert "states" in item["metadata"]
        # assert "selected_rules" in item["metadata"]
        assert "solution" in item["metadata"]

        string = item["metadata"]["string"]
        solution = item["metadata"]["solution"]
        states = item["metadata"]["states"]
        selected_rules = item["metadata"]["selected_rules"]

        # Verify dimensions
        assert config.min_string_length <= len(string) <= config.max_string_length
        assert config.min_num_rules <= len(selected_rules) <= config.max_num_rules
        assert len(states) >= 1
        assert solution == states[-1]


def test_string_manipulation_dataset_iteration():
    """Test that iteration respects dataset size"""
    config = StringManipulationConfig(size=5, seed=42)
    dataset = StringManipulationDataset(config)

    items = list(dataset)
    assert len(items) == config.size

    # Test multiple iterations yield same items
    assert items == list(dataset)


def test_string_manipulation_answer():
    """Test the method for getting the answer"""
    config = StringManipulationConfig(seed=42)
    dataset = StringManipulationDataset(config)

    rules = [
        (
            "If the string prefix is 'ab', replace it with 'ca'.",
            lambda s: ("ca" + s[2:], 1) if s.startswith("ab") else (s, 0),
        )
    ]
    assert dataset._get_all_transforms("abbbab", rules)[-1] == "cabbab"

    rules = [
        (
            "If the string suffix is 'ac', replace it with 'cb'.",
            lambda s: (s[:-2] + "cb", 2) if s.endswith("ac") else (s, 0),
        ),
    ]
    assert dataset._get_all_transforms("abbbac", rules)[-1] == "abbbcb"

    rules = [
        (
            "If the string prefix is 'bc', delete the first two characters and append 'aa' to the end.",
            lambda s: (s[2:] + "aa", 3) if s.startswith("bc") else (s, 0),
        ),
    ]
    assert dataset._get_all_transforms("bcabbb", rules)[-1] == "abbbaa"

    rules = [
        (
            "If the string suffix is 'bb', delete the last two characters.",
            lambda s: (s[:-2], 4) if s.endswith("bb") else (s, 0),
        ),
    ]
    assert dataset._get_all_transforms("abbbabb", rules)[-1] == "abbba"

    rules = [
        (
            "If the string prefix is 'cb', replace it with 'aa' and delete the last character.",
            lambda s: ("aa" + s[2:-1], 5) if s.startswith("cb") and len(s) > 1 else (s, 0),
        )
    ]
    assert dataset._get_all_transforms("cbabbb", rules)[-1] == "aaabb"

    rules = [
        (
            "If the string prefix is 'ca', replace it with 'bb' and append 'c' to the end.",
            lambda s: ("bb" + s[2:] + "c", 6) if s.startswith("ca") else (s, 0),
        )
    ]
    assert dataset._get_all_transforms("caabbb", rules)[-1] == "bbabbbc"

    rules = [
        (
            "If the string suffix is 'cc', replace it with 'b' and prepend 'a' to the start.",
            lambda s: ("a" + s[:-2] + "b", 7) if s.endswith("cc") else (s, 0),
        )
    ]
    assert dataset._get_all_transforms("abbbcc", rules)[-1] == "aabbbb"

    rules = [
        (
            "If the string prefix is 'aa', remove the first character.",
            lambda s: (s[1:], 8) if s.startswith("aa") else (s, 0),
        )
    ]
    assert dataset._get_all_transforms("aabbb", rules)[-1] == "abbb"

    rules = [
        (
            "If the string contains 'abc', replace the first occurrence with 'cab'.",
            lambda s: (s.replace("abc", "cab", 1), 9) if "abc" in s else (s, 0),
        )
    ]
    assert dataset._get_all_transforms("ababcb", rules)[-1] == "cababb"  # 'ababcb' -> 'abcabb' -> 'cababb'

    rules = [
        (
            "If the string contains 'bca', delete the first occurrence entirely.",
            lambda s: (s.replace("bca", "", 1), 10) if "bca" in s else (s, 0),
        )
    ]
    assert dataset._get_all_transforms("abbcab", rules)[-1] == "abb"

    rules = [
        (
            "If the string ends with 'ba', replace it with 'ab'.",
            lambda s: (s[:-2] + "ab", 11) if s.endswith("ba") else (s, 0),
        )
    ]
    assert dataset._get_all_transforms("abbbba", rules)[-1] == "abbbab"

    rules = [
        (
            "If the string starts with 'cc', remove the first two characters.",
            lambda s: (s[2:], 12) if s.startswith("cc") else (s, 0),
        )
    ]
    assert dataset._get_all_transforms("ccabbb", rules)[-1] == "abbb"

    rules = [
        (
            "If the string contains 'acb', replace the first occurrence with its reverse ('bca').",
            lambda s: (s.replace("acb", "bca", 1), 13) if "acb" in s else (s, 0),
        )
    ]
    assert dataset._get_all_transforms("abacbb", rules)[-1] == "abbcab"

    rules = [
        (
            "If the string contains 'acb', replace the first occurrence with its reverse ('bca').",
            lambda s: (s.replace("acb", "bca", 1), 13) if "acb" in s else (s, 0),
        )
    ]
    assert dataset._get_all_transforms("abacbb", rules)[-1] == "abbcab"

    rules = [
        (
            "If the string ends with 'ca', remove the last character.",
            lambda s: (s[:-1], 14) if s.endswith("ca") and len(s) > 0 else (s, 0),
        )
    ]
    assert dataset._get_all_transforms("abbbca", rules)[-1] == "abbbc"

    rules = [
        (
            "If the string starts with 'bb', remove the second character.",
            lambda s: (s[0] + s[2:], 15) if s.startswith("bb") and len(s) >= 2 else (s, 0),
        )
    ]
    assert dataset._get_all_transforms("bbabcbb", rules)[-1] == "babcbb"

    rules = [
        (
            "If the string ends with 'aa', replace it with 'cc'.",
            lambda s: (s[:-2] + "cc", 16) if s.endswith("aa") else (s, 0),
        )
    ]
    assert dataset._get_all_transforms("abccbaa", rules)[-1] == "abccbcc"

    rules = [
        (
            "If the string contains 'ca' (not at the start), remove the first occurrence found after the first character.",
            lambda s: (s[:idx] + s[idx + 2 :], 17) if (idx := s.find("ca", 1)) != -1 else (s, 0),
        )
    ]
    assert dataset._get_all_transforms("abacab", rules)[-1] == "abab"
    assert dataset._get_all_transforms("caabab", rules)[-1] == "caabab"

    rules = [
        (
            "If the string contains an even number of 'b's (and at least one 'b'), append 'ab' at the end.",
            lambda s: (s + "ab", 18) if (s.count("b") > 0 and s.count("b") % 2 == 0) else (s, 0),
        )
    ]
    assert dataset._get_all_transforms("abab", rules)[-1] == "ababab"
    assert dataset._get_all_transforms("abbab", rules)[-1] == "abbab"

    rules = [
        (
            "If the string length is greater than 15, remove the middle character.",
            lambda s: (s[: len(s) // 2] + s[len(s) // 2 + 1 :], 19) if len(s) > 15 else (s, 0),
        )
    ]
    assert (
        dataset._get_all_transforms("bccbcbbbcbbbbcccc", rules)[-1] == "bccbcbbbbbbcccc"
    )  # bccbcbbbcbbbbcccc -> "bccbcbbbbbbbcccc" -> "bccbcbbbbbbcccc"

    rules = [
        (
            "If the string starts with 'ac', replace the first two characters with 'zz'.",
            lambda s: ("zz" + s[2:], 20) if s.startswith("ac") else (s, 0),
        )
    ]
    assert dataset._get_all_transforms("acab", rules)[-1] == "zzab"
