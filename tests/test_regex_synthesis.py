"""Tests for regex synthesis task generation, scoring and the safe regex engine"""

import itertools
import re
import time
from random import Random

import pytest

import reasoning_gym
from reasoning_gym.induction.regex_synthesis import (
    RegexSynthesisConfig,
    RegexSynthesisCurriculum,
    RegexSynthesisDataset,
    _normalize_answer,
)
from reasoning_gym.induction.safe_regex import MAX_PATTERN_LENGTH, UnsupportedRegexError, compile_safe_regex


def _make_entry(positives, negatives, hidden_positives, hidden_negatives, answer="ab*"):
    return {
        "answer": answer,
        "metadata": {
            "positive_examples": positives,
            "negative_examples": negatives,
            "hidden_positives": hidden_positives,
            "hidden_negatives": hidden_negatives,
        },
    }


# entry for the target ab*c over the alphabet {a, b, c}
AB_STAR_C = _make_entry(
    positives=["ac", "abc", "abbc"],
    negatives=["", "a", "bc", "acc"],
    hidden_positives=["abbbc", "abbbbc"],
    hidden_negatives=["ab", "cab", "abcc", "aabc"],
    answer="ab*c",
)


# ------------------------------------------------------------------ config


def test_regex_synthesis_config_validation():
    """Test that invalid configs raise appropriate errors"""
    bad_configs = [
        dict(min_components=0),
        dict(min_components=5, max_components=4),
        dict(max_components=13),
        dict(alphabet_size=1),
        dict(alphabet_size=27),
        dict(operator_level=0),
        dict(operator_level=4),
        dict(num_examples=0),
        dict(num_hidden=0),
        dict(max_string_length=3),
    ]
    for kwargs in bad_configs:
        with pytest.raises(AssertionError):
            RegexSynthesisConfig(**kwargs).validate()


# ------------------------------------------------------------------ generation


def test_regex_synthesis_deterministic():
    """Test that dataset generates same items with same seed"""
    config = RegexSynthesisConfig(seed=42, size=20)
    dataset1 = RegexSynthesisDataset(config)
    dataset2 = RegexSynthesisDataset(config)
    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_regex_synthesis_iteration():
    """Test that iteration respects dataset size"""
    config = RegexSynthesisConfig(size=5, seed=42)
    dataset = RegexSynthesisDataset(config)
    items = list(dataset)
    assert len(items) == config.size
    assert items == list(dataset)


@pytest.mark.parametrize("operator_level", [1, 2, 3])
@pytest.mark.parametrize("alphabet_size", [2, 3, 5])
def test_regex_synthesis_items(operator_level, alphabet_size):
    """Test item structure and that all strings are labelled correctly (checked with Python's re)"""
    config = RegexSynthesisConfig(
        operator_level=operator_level,
        alphabet_size=alphabet_size,
        min_components=2,
        max_components=5,
        size=40,
        seed=1234,
    )
    dataset = RegexSynthesisDataset(config)
    alphabet = set("abcdefghijklmnopqrstuvwxyz"[:alphabet_size])

    for item in dataset:
        assert set(item.keys()) == {"question", "answer", "metadata"}
        meta = item["metadata"]
        target = re.compile(item["answer"])
        assert meta["target_regex"] == item["answer"]
        assert config.min_components <= meta["num_components"] <= config.max_components

        positives = meta["positive_examples"] + meta["hidden_positives"]
        negatives = meta["negative_examples"] + meta["hidden_negatives"]
        assert len(meta["positive_examples"]) == len(meta["negative_examples"]) == config.num_examples
        assert len(meta["hidden_positives"]) == len(meta["hidden_negatives"]) == config.num_hidden

        # labels are correct according to Python's own regex engine
        assert all(target.fullmatch(s) for s in positives)
        assert not any(target.fullmatch(s) for s in negatives)

        # no duplicates, no overlap between shown and hidden strings
        all_strings = positives + negatives
        assert len(set(all_strings)) == len(all_strings)

        # strings respect the alphabet and length limit
        for s in all_strings:
            assert set(s) <= alphabet
            assert len(s) <= config.max_string_length

        # every shown string appears in the prompt, hidden strings do not
        for s in meta["positive_examples"] + meta["negative_examples"]:
            assert f'"{s}"' in item["question"]
        for s in meta["hidden_positives"] + meta["hidden_negatives"]:
            assert f'"{s}"\n' not in item["question"]

        assert 0.0 <= meta["feature_coverage"] <= 1.0


def test_regex_synthesis_feature_coverage():
    """Shown positives are chosen to demonstrate the choices used by hidden positives"""
    dataset = RegexSynthesisDataset(RegexSynthesisConfig(operator_level=3, max_components=4, size=100, seed=7))
    coverage = [item["metadata"]["feature_coverage"] for item in dataset]
    assert sum(coverage) / len(coverage) > 0.9


def test_regex_synthesis_reference_answer_scores_one():
    dataset = RegexSynthesisDataset(RegexSynthesisConfig(operator_level=3, max_components=6, size=50, seed=99))
    for item in dataset:
        assert dataset.score_answer(item["answer"], item) == 1.0


def test_regex_synthesis_create_dataset():
    """Test registration via reasoning_gym.create_dataset"""
    dataset = reasoning_gym.create_dataset("regex_synthesis", size=5, seed=1)
    assert isinstance(dataset, RegexSynthesisDataset)
    for item in dataset:
        assert dataset.score_answer(item["answer"], item) == 1.0


# ------------------------------------------------------------------ scoring


def test_regex_synthesis_score_equivalent_answers():
    """Any regex describing the same language is fully correct"""
    dataset = RegexSynthesisDataset(RegexSynthesisConfig(seed=1, size=1))
    for answer in ["ab*c", "a(b)*c", "ab{0,}c", "a(?:b*)c", "^ab*c$", "a[b]*c", "a(b|bb)*c", "ab*?c"]:
        assert dataset.score_answer(answer, AB_STAR_C) == 1.0, answer


def test_regex_synthesis_score_wrappers():
    """Common answer formatting is tolerated"""
    dataset = RegexSynthesisDataset(RegexSynthesisConfig(seed=1, size=1))
    for answer in ["  ab*c\n", "`ab*c`", "/ab*c/", "```\nab*c\n```", "```regex\nab*c\n```"]:
        assert dataset.score_answer(answer, AB_STAR_C) == 1.0, answer


def test_regex_synthesis_score_wrong_answers():
    dataset = RegexSynthesisDataset(RegexSynthesisConfig(seed=1, size=1))
    wrong = [
        None,
        "",
        "   ",
        ".*",  # matches the shown negatives
        "ac|abc|abbc",  # overfits to the shown positives
        "a[bc]*",  # matches the shown negative "acc"
        "(((",  # syntax error
        "a(b)\\1c",  # back-reference
        "(?=a)ab*c",  # lookahead
        "ab*+c",  # possessive quantifier
        "__import__('os').system('ls')",  # nonsense / injection attempt is just a non-matching regex
        "a" * (MAX_PATTERN_LENGTH + 1),
        123,
    ]
    for answer in wrong:
        assert dataset.score_answer(answer, AB_STAR_C) == 0.0, answer


def test_regex_synthesis_score_partial():
    """A regex that fits the shown examples but not all hidden strings gets partial credit"""
    dataset = RegexSynthesisDataset(RegexSynthesisConfig(seed=1, size=1))
    # ab{0,2}c fits all shown examples, rejects both hidden positives and all hidden negatives -> 0 * 1
    assert dataset.score_answer("ab{0,2}c", AB_STAR_C) == 0.0
    # ab{0,3}c matches one of the two hidden positives -> 0.5 * 1
    assert dataset.score_answer("ab{0,3}c", AB_STAR_C) == pytest.approx(0.5)
    # ab*c|cab also matches the hidden negative "cab" -> 1 * 0.75
    assert dataset.score_answer("ab*c|cab", AB_STAR_C) == pytest.approx(0.75)


def test_regex_synthesis_score_is_fast_on_pathological_answers():
    """Catastrophic-backtracking patterns cannot hang the scorer"""
    entry = _make_entry(
        positives=["a" * 10 + "b"],
        negatives=["a" * 10],
        hidden_positives=["a" * 16 + "b"],
        hidden_negatives=["a" * 5000, "a" * 4999 + "c"],
    )
    dataset = RegexSynthesisDataset(RegexSynthesisConfig(seed=1, size=1))
    start = time.time()
    for answer in ["(a+)+b", "(a*)*b", "(a|aa)+b", "(a|a?)+b"]:
        assert dataset.score_answer(answer, entry) == 1.0
    assert time.time() - start < 5.0


def test_normalize_answer():
    assert _normalize_answer(" ab ") == "ab"
    assert _normalize_answer("`ab`") == "ab"
    assert _normalize_answer("/ab/") == "ab"
    assert _normalize_answer("```python\nab\n```") == "ab"
    assert _normalize_answer("x" * 10_000) is None


# ------------------------------------------------------------------ safe regex engine

ENGINE_PATTERNS = [
    "",
    "a",
    "abc",
    "a*b",
    "a+b?",
    "(ab|c)+",
    "(a|)b",
    "a|b|",
    "()*",
    "(a*)*b",
    "[a-c]{2,3}",
    "[^ab]*",
    "a{,2}b",
    "a{2,}",
    "a{0}",
    "a{3}",
    "x?y??z*?",
    "(?:ab)*c",
    "\\d+\\.\\d",
    "[\\w-]+",
    "[]a]+",
    "[a-]*",
    "a.c",
    "^ab$",
    "\\$a",
    "[ab]c|d",
    "((a|b)c)*",
    "[^\\d]b",
]
ENGINE_ALPHABET = "abc1.-]$"


@pytest.mark.parametrize("pattern", ENGINE_PATTERNS)
def test_safe_regex_agrees_with_re(pattern):
    """The safe engine gives the same full-match result as Python's re on all short strings"""
    ours = compile_safe_regex(pattern)
    theirs = re.compile(pattern)
    strings = [""] + ["".join(p) for n in range(1, 4) for p in itertools.product(ENGINE_ALPHABET, repeat=n)]
    for s in strings:
        assert ours.fullmatch(s) == bool(theirs.fullmatch(s)), (pattern, s)


def test_safe_regex_agrees_with_re_on_generated_targets():
    """Differential test on the target regexes the dataset actually produces"""
    rng = Random(0)
    dataset = RegexSynthesisDataset(RegexSynthesisConfig(operator_level=3, max_components=6, size=100, seed=5))
    for item in dataset:
        ours = compile_safe_regex(item["answer"])
        theirs = re.compile(item["answer"])
        alphabet = item["metadata"]["alphabet"]
        for _ in range(50):
            s = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 10)))
            assert ours.fullmatch(s) == bool(theirs.fullmatch(s)), (item["answer"], s)


@pytest.mark.parametrize(
    "pattern",
    [
        "(a)\\1",  # back-reference
        "(?=a)a",  # lookahead
        "(?<=a)b",  # lookbehind
        "(?i)a",  # inline flag
        "(?P<x>a)",  # named group
        "a**",  # multiple repeat
        "a*+",  # possessive
        "a?*",
        "\\b",
        "\\Z",
        "a$b",  # anchor in the middle
        "a^b",
        "(a",
        "a)",
        "[a",
        "[z-a]",
        "*a",
        "{",
        "a{3,1}",
        "a{21}",
        "a{1,99}",
        "\\",
        "((a{20}){20}){20}",  # too many NFA states
        "a" * (MAX_PATTERN_LENGTH + 1),
    ],
)
def test_safe_regex_rejects_unsupported(pattern):
    with pytest.raises(UnsupportedRegexError):
        compile_safe_regex(pattern)


def test_safe_regex_linear_time():
    start = time.time()
    assert not compile_safe_regex("(a+)+b").fullmatch("a" * 20_000)
    assert not compile_safe_regex("(a*)*(a|aa)*c").fullmatch("a" * 20_000 + "b")
    assert compile_safe_regex("(a|aa)*").fullmatch("a" * 20_000)
    assert time.time() - start < 5.0


# ------------------------------------------------------------------ curriculum


def test_regex_synthesis_curriculum():
    curriculum = RegexSynthesisCurriculum()
    base_value = {"size": 150, "seed": 1}

    base_cfg: RegexSynthesisConfig = curriculum.generate_configuration(base_value)
    assert base_cfg.seed == 1
    assert base_cfg.size == 150
    assert base_cfg.min_components == 2 and base_cfg.max_components == 2
    assert base_cfg.operator_level == 1
    assert base_cfg.alphabet_size == 2

    curriculum.increment_attr_level("num_components")
    curriculum.increment_attr_level("operator_level")
    curriculum.increment_attr_level("alphabet_size")
    increased_cfg = curriculum.generate_configuration(base_value)
    assert increased_cfg.min_components == 2 and increased_cfg.max_components == 3
    assert increased_cfg.operator_level == 2
    assert increased_cfg.alphabet_size == 3

    curriculum.decrement_attr_level("num_components")
    decreased_cfg = curriculum.generate_configuration(base_value)
    assert decreased_cfg.min_components == 2 and decreased_cfg.max_components == 2


def test_regex_synthesis_curriculum_all_levels_generate():
    """Every curriculum level produces valid, self-consistent items"""
    curriculum = RegexSynthesisCurriculum()
    for level in range(4):
        curriculum.set_global_level(level)
        config = curriculum.generate_configuration({"size": 20, "seed": level})
        dataset = RegexSynthesisDataset(config)
        for item in dataset:
            assert dataset.score_answer(item["answer"], item) == 1.0
