"""Tests for the Codebreaker reasoning game."""

import pytest

from reasoning_gym.games.codebreaker import (
    CodebreakerConfig,
    CodebreakerDataset,
    get_feedback,
)


def test_feedback_counts_exact_positions():
    secret = ("RED", "BLUE", "GREEN", "YELLOW")
    guess = ("RED", "BLUE", "PURPLE", "ORANGE")

    assert get_feedback(secret, guess) == (2, 0)


def test_feedback_counts_misplaced_colors():
    secret = ("RED", "BLUE", "GREEN", "YELLOW")
    guess = ("BLUE", "RED", "YELLOW", "GREEN")

    assert get_feedback(secret, guess) == (0, 4)


def test_feedback_handles_repeated_colors():
    secret = ("RED", "RED", "BLUE", "GREEN")
    guess = ("RED", "BLUE", "RED", "YELLOW")

    assert get_feedback(secret, guess) == (1, 2)


def test_dataset_entry_has_required_keys():
    dataset = CodebreakerDataset(
        CodebreakerConfig(seed=42),
        size=5,
    )

    entry = dataset[0]

    assert "question" in entry
    assert "answer" in entry
    assert "metadata" in entry
    assert isinstance(entry["question"], str)
    assert isinstance(entry["answer"], str)
    assert isinstance(entry["metadata"], dict)


def test_same_seed_is_reproducible():
    dataset1 = CodebreakerDataset(
        CodebreakerConfig(seed=42),
        size=5,
    )
    dataset2 = CodebreakerDataset(
        CodebreakerConfig(seed=42),
        size=5,
    )

    assert dataset1[0]["answer"] == dataset2[0]["answer"]
    assert dataset1[0]["question"] == dataset2[0]["question"]


def test_different_indices_generate_entries():
    dataset = CodebreakerDataset(
        CodebreakerConfig(seed=42),
        size=5,
    )

    assert dataset[0]["answer"] != dataset[1]["answer"]


def test_correct_answer_receives_full_score():
    dataset = CodebreakerDataset(
        CodebreakerConfig(seed=42),
        size=5,
    )
    entry = dataset[0]

    assert dataset.score_answer(entry["answer"], entry) == 1.0


def test_incorrect_answer_receives_zero_score():
    dataset = CodebreakerDataset(
        CodebreakerConfig(seed=42),
        size=5,
    )
    entry = dataset[0]

    assert dataset.score_answer("NOT A VALID CODE", entry) == 0.0


def test_answer_is_case_insensitive():
    dataset = CodebreakerDataset(
        CodebreakerConfig(seed=42),
        size=5,
    )
    entry = dataset[0]

    assert dataset.score_answer(entry["answer"].lower(), entry) == 1.0


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError):
        CodebreakerConfig(code_length=1).validate()


def test_invalid_index_raises_error():
    dataset = CodebreakerDataset(
        CodebreakerConfig(seed=42),
        size=5,
    )

    with pytest.raises(IndexError):
        dataset[5]