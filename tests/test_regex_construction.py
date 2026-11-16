import random
import re

import pytest

from reasoning_gym.algorithmic.regex_construction import (
    MAX_ANSWER_LENGTH,
    RegexConstructionConfig,
    RegexConstructionCurriculum,
    RegexConstructionDataset,
)


def test_regex_construction_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        RegexConstructionConfig(alphabet="ab").validate()  # too few characters

    with pytest.raises(AssertionError):
        RegexConstructionConfig(alphabet="abca").validate()  # duplicates

    with pytest.raises(AssertionError):
        RegexConstructionConfig(alphabet="abc.").validate()  # regex metacharacter

    with pytest.raises(AssertionError):
        RegexConstructionConfig(min_length=1).validate()

    with pytest.raises(AssertionError):
        RegexConstructionConfig(min_length=8, max_length=5).validate()

    with pytest.raises(AssertionError):
        RegexConstructionConfig(max_length=MAX_ANSWER_LENGTH).validate()

    with pytest.raises(AssertionError):
        RegexConstructionConfig(min_patterns=0).validate()

    with pytest.raises(AssertionError):
        RegexConstructionConfig(min_forbidden=2, max_forbidden=1).validate()


def test_regex_construction_deterministic():
    """Test that dataset generates same items with same seed"""
    config = RegexConstructionConfig(seed=42, size=20)
    dataset1 = RegexConstructionDataset(config)
    dataset2 = RegexConstructionDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_regex_construction_items():
    """Test basic properties of generated items"""
    config = RegexConstructionConfig(min_length=4, max_length=12, max_patterns=4, max_forbidden=3, size=100, seed=42)
    dataset = RegexConstructionDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        metadata = item["metadata"]
        assert metadata["source_dataset"] == "regex_construction"
        assert 4 <= metadata["length"] <= 12
        assert 2 <= len(metadata["patterns"]) <= 4
        assert 0 <= len(metadata["forbidden"]) <= 3

        target = item["answer"]
        assert len(target) == metadata["length"]
        assert all(c in config.alphabet for c in target)
        for p in metadata["patterns"]:
            assert re.fullmatch(p, target), f"{target} should match {p}"
            assert p in item["question"]
        for p in metadata["forbidden"]:
            assert not re.fullmatch(p, target), f"{target} should not match {p}"
            assert p in item["question"]

        assert target not in metadata["patterns"], "a required pattern should never just spell out the answer"
        assert dataset.score_answer(target, entry=item) == 1.0


def test_regex_construction_patterns_stay_cheap():
    """Patterns should have few open-ended quantifiers and never nest them"""
    config = RegexConstructionConfig(min_length=20, max_length=40, max_patterns=4, max_forbidden=3, size=100, seed=3)
    dataset = RegexConstructionDataset(config)

    for item in dataset:
        for p in item["metadata"]["patterns"] + item["metadata"]["forbidden"]:
            assert p.count("*") + p.count("+") <= 2
            assert not re.search(r"\)[*+?{]", p), f"quantified group in {p}"


def test_regex_construction_forbidden_count():
    config = RegexConstructionConfig(min_forbidden=2, max_forbidden=2, size=30, seed=11)
    dataset = RegexConstructionDataset(config)

    for item in dataset:
        assert len(item["metadata"]["forbidden"]) == 2
        assert "must not fully match" in item["question"]

    config = RegexConstructionConfig(min_forbidden=0, max_forbidden=0, size=10, seed=11)
    for item in RegexConstructionDataset(config):
        assert item["metadata"]["forbidden"] == []
        assert "must not fully match" not in item["question"]


def test_regex_construction_score_answer():
    dataset = RegexConstructionDataset(RegexConstructionConfig(size=1, seed=1))
    entry = {
        "question": "",
        "answer": "acb24",
        "metadata": {
            "alphabet": "abcdef012345",
            "patterns": [r"[a-c]{3}[0-5]+", r"(ca|ac)b.*", r".{2}b[2-5]{2}"],
            "forbidden": [r"ca.*"],
        },
    }

    assert dataset.score_answer("acb24", entry) == 1.0
    assert dataset.score_answer("acb33", entry) == 1.0  # any valid string counts, not just the stored one
    assert dataset.score_answer("  acb24\n", entry) == 1.0
    assert dataset.score_answer('"acb24"', entry) == 1.0
    assert dataset.score_answer("`acb24`", entry) == 1.0

    assert dataset.score_answer("cab42", entry) == 0.0  # matches all required patterns but also a forbidden one
    assert dataset.score_answer("acb2", entry) == 0.0  # too short for the third pattern
    assert dataset.score_answer("acb2x", entry) == 0.0
    assert dataset.score_answer("ACB24", entry) == 0.0  # outside the alphabet
    assert dataset.score_answer("acb 24", entry) == 0.0
    assert dataset.score_answer("", entry) == 0.0
    assert dataset.score_answer(None, entry) == 0.0
    assert dataset.score_answer("a" * (MAX_ANSWER_LENGTH + 1), entry) == 0.0


def test_regex_construction_alternative_answers():
    """Strings other than the stored answer are scored by the constraints alone"""
    config = RegexConstructionConfig(alphabet="abc01", min_length=3, max_length=5, size=30, seed=5)
    dataset = RegexConstructionDataset(config)
    rng = random.Random(0)

    found_alternative = False
    for item in dataset:
        metadata = item["metadata"]
        for _ in range(300):
            s = "".join(rng.choice(config.alphabet) for _ in range(metadata["length"]))
            expected = all(re.fullmatch(p, s) for p in metadata["patterns"]) and not any(
                re.fullmatch(p, s) for p in metadata["forbidden"]
            )
            assert dataset.score_answer(s, entry=item) == float(expected)
            if expected and s != item["answer"]:
                found_alternative = True

    assert found_alternative


def test_regex_construction_hard_to_guess():
    """A random string of the right length should almost never pass"""
    config = RegexConstructionConfig(size=100, seed=8)
    dataset = RegexConstructionDataset(config)
    rng = random.Random(0)

    total = 0.0
    for item in dataset:
        for _ in range(50):
            s = "".join(rng.choice(config.alphabet) for _ in range(item["metadata"]["length"]))
            total += dataset.score_answer(s, entry=item)

    assert total / (100 * 50) < 0.01


def test_regex_construction_curriculum():
    curriculum = RegexConstructionCurriculum()

    base_value = {"size": 150, "seed": 1}

    base_cfg: RegexConstructionConfig = curriculum.generate_configuration(base_value)
    assert base_cfg.seed == 1
    assert base_cfg.size == 150
    assert base_cfg.min_length == 5 and base_cfg.max_length == 10
    assert base_cfg.min_patterns == 1 and base_cfg.max_patterns == 1
    assert base_cfg.min_forbidden == 0 and base_cfg.max_forbidden == 0

    # test incrementing attribute levels
    curriculum.increment_attr_level("length")
    curriculum.increment_attr_level("num_patterns")
    curriculum.increment_attr_level("num_forbidden")
    increased_cfg = curriculum.generate_configuration(base_value)
    assert increased_cfg.min_length == 5 and increased_cfg.max_length == 15
    assert increased_cfg.min_patterns == 1 and increased_cfg.max_patterns == 2
    assert increased_cfg.min_forbidden == 0 and increased_cfg.max_forbidden == 1

    # test decrementing attribute levels
    curriculum.decrement_attr_level("length")
    partially_decreased_cfg = curriculum.generate_configuration(base_value)
    assert partially_decreased_cfg.min_length == 5 and partially_decreased_cfg.max_length == 10
    assert partially_decreased_cfg.max_patterns == 2

    # the hardest level should still generate valid puzzles
    curriculum.set_global_level(4)
    hard_cfg = curriculum.generate_configuration(base_value)
    assert hard_cfg.max_length == 30 and hard_cfg.max_patterns == 4 and hard_cfg.max_forbidden == 3
    dataset = RegexConstructionDataset(hard_cfg)
    for i in range(20):
        item = dataset[i]
        assert dataset.score_answer(item["answer"], entry=item) == 1.0
