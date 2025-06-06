"""
Tests for the get_score_answer_fn helper with hard-coded sample cases.
"""

import pytest

from reasoning_gym import get_score_answer_fn

TEST_CASES = [
    {
        "dataset": "letter_jumble",
        "entry": {"answer": "second opportunity to receive"},
        "model_answer": "second opportunity to receive",
        "expected": 1.0,
        "id": "rg_4806-correct",
    },
    {
        "dataset": "word_sorting",
        "entry": {
            "answer": "arrive, burdens, computers, federal, louder, paragraphs, side, specified, virus",
            "metadata": {
                "sorted_words": [
                    "arrive",
                    "burdens",
                    "computers",
                    "federal",
                    "louder",
                    "paragraphs",
                    "side",
                    "specified",
                    "virus",
                ]
            },
        },
        "model_answer": "arrive, burdens, computers, federal, louder, paragraphs, side, specified, virus",
        "expected": 1.0,
        "id": "rg_16004-word_sorting-correct",
    },
    {
        "dataset": "spell_backward",
        "entry": {"answer": "ssiknu"},
        "model_answer": "ssiknu",
        "expected": 1.0,
        "id": "rg_14211-correct",
    },
    {
        "dataset": "letter_jumble",
        "entry": {"answer": "second opportunity to receive"},
        "model_answer": "completely wrong answer here",
        "expected": 0.0,
        "id": "rg_4806-incorrect",
    },
    {
        "dataset": "spell_backward",
        "entry": {"answer": "ssiknu"},
        "model_answer": "unkiss",
        "expected": 0.0,
        "id": "rg_14211-incorrect",
    },
]


@pytest.mark.parametrize("case", TEST_CASES, ids=lambda c: c["id"])
def test_get_score_answer_fn_hardcoded(case):
    """
    Ensure the dataset-specific scorer returns the expected value
    for the given model answer and entry.
    """
    scorer = get_score_answer_fn(case["dataset"])
    returned = scorer(case["model_answer"], case["entry"])
    assert returned == pytest.approx(case["expected"], abs=1e-8)
