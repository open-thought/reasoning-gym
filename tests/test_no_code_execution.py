"""Regression tests: the four literal-fallback score_answer sites must never execute code in an answer."""

import pytest

import reasoning_gym

# The four datasets whose score_answer falls back to parsing the answer as a Python literal.
LITERAL_FALLBACK_DATASETS = ["binary_matrix", "spiral_matrix", "string_insertion", "n_queens"]


@pytest.mark.parametrize("dataset_name", LITERAL_FALLBACK_DATASETS)
def test_score_answer_does_not_execute_code(dataset_name, tmp_path):
    """A candidate answer that would write a file must score 0.0 and leave no file behind."""
    marker = tmp_path / "pwned"
    candidate = f"__import__('pathlib').Path({str(marker)!r}).write_text('x')"

    dataset = reasoning_gym.create_dataset(dataset_name, seed=42, size=1)
    entry = dataset[0]
    score = dataset.score_answer(candidate, entry)

    assert not marker.exists(), f"{dataset_name}.score_answer executed the candidate answer"
    assert score == 0.0


@pytest.mark.parametrize("dataset_name", LITERAL_FALLBACK_DATASETS)
def test_score_answer_still_accepts_python_literal(dataset_name):
    """The literal fallback (partial credit for a Python list form of the answer) keeps working."""
    dataset = reasoning_gym.create_dataset(dataset_name, seed=42, size=1)
    entry = dataset[0]
    if dataset_name == "string_insertion":
        literal = repr(list(entry["answer"]))
        expected = 0.1
    elif dataset_name == "spiral_matrix":
        literal = repr([int(x) for x in entry["answer"].split()])
        expected = 0.1
    elif dataset_name == "binary_matrix":
        literal = repr([[int(x) for x in row.split()] for row in entry["answer"].split("\n")])
        expected = 0.1
    else:  # n_queens
        literal = repr([row.split() for row in entry["metadata"]["valid_answers"][0].split("\n")])
        expected = 0.5
    assert dataset.score_answer(literal, entry) == expected
