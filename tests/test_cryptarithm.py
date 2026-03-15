import pytest

from reasoning_gym import create_dataset
from reasoning_gym.algorithmic.cryptarithm import (
    CryptarithmConfig,
    CryptarithmCurriculum,
    CryptarithmDataset,
    verify_cryptarithm_solution,
)


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


def test_cryptarithm_score_answer():
    """Test the CryptarithmDataset.score_answer method for various correctness levels."""
    dataset = create_dataset("cryptarithm", seed=42, size=1)
    puzzle = dataset[0]
    correct_answer_str = puzzle["answer"]  # e.g. "A=1,B=7,..."

    # 1) Correct mapping => expecting 1.0
    score = dataset.score_answer(answer=correct_answer_str, entry=puzzle)
    assert score == 1.0, f"Expected 1.0 for perfectly correct answer, got {score}"

    # 2) Correct mapping in different order => should still be 1.0
    correct_mapping = {}
    for pair in correct_answer_str.split(","):
        alpha, num_str = pair.split("=")
        correct_mapping[alpha] = int(num_str)
    reversed_answer = ",".join(
        f"{letter}={correct_mapping[letter]}" for letter in reversed(sorted(correct_mapping.keys()))
    )
    score = dataset.score_answer(answer=reversed_answer, entry=puzzle)
    assert score == 1.0, f"Expected 1.0 for correct answer in different order, got {score}"

    # 3) Mismatch number of pairs => score should be 0.0 (parse succeeds but validation fails)
    # For instance, drop the last pair
    splitted = correct_answer_str.split(",")
    mismatch_str = ",".join(splitted[:-1])
    score = dataset.score_answer(answer=mismatch_str, entry=puzzle)
    assert score == 0.01, f"Expected 0.01 when #pairs does not match (missing letter), got {score}"

    # 4) Parse error => 0.0 (e.g. remove '=' from the first pair)
    splitted = correct_answer_str.split(",")
    splitted[0] = splitted[0].replace("=", "")  # remove '=' in the first pair
    parse_error_str = ",".join(splitted)
    score = dataset.score_answer(answer=parse_error_str, entry=puzzle)
    assert score == 0.0, f"Expected 0.0 when parsing fails on at least one pair, got {score}"

    # 5) Correct number of pairs, but duplicate alphabets => 0.01 (parseable but invalid)
    # This makes the dictionary have fewer unique keys than expected
    splitted = correct_answer_str.split(",")
    if len(splitted) > 1:
        splitted[0] = splitted[1]  # Duplicate the second pair in the first position
    duplicates_str = ",".join(splitted)
    score = dataset.score_answer(answer=duplicates_str, entry=puzzle)
    assert score == 0.01, f"Expected 0.01 if the final dict has fewer unique alphabets, got {score}"

    # 6) Wrong arithmetic - swap two digits to break the equation
    correct_mapping = {}
    for pair in correct_answer_str.split(","):
        alpha, num_str = pair.split("=")
        correct_mapping[alpha] = int(num_str)

    # Swap two digit assignments to break arithmetic
    letters = list(correct_mapping.keys())
    if len(letters) >= 2:
        wrong_mapping = correct_mapping.copy()
        wrong_mapping[letters[0]], wrong_mapping[letters[1]] = (
            wrong_mapping[letters[1]],
            wrong_mapping[letters[0]],
        )

        wrong_answer_str = ",".join(f"{l}={wrong_mapping[l]}" for l in sorted(letters))
        score = dataset.score_answer(answer=wrong_answer_str, entry=puzzle)
        assert score == 0.01, f"Expected 0.01 for invalid arithmetic, got {score}"

    # 7) None or non-string answer => 0.0
    score = dataset.score_answer(answer=None, entry=puzzle)
    assert score == 0.0, f"Expected 0.0 for None answer, got {score}"


def test_cryptarithm_verify_solution():
    """Test the verify_cryptarithm_solution helper function."""

    # Test case 1: Valid solution with simple arithmetic
    mapping = {"A": 1, "B": 2}
    words = ["A", "A"]  # 1 + 1
    result = "B"  # 2
    is_valid, reason = verify_cryptarithm_solution(mapping, words, result, True)
    assert is_valid, f"Valid solution marked invalid: {reason}"

    # Test case 2: Valid solution with multi-digit numbers
    mapping = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 7}
    words = ["AB", "CE"]  # 12 + 35
    result = "DF"  # 47
    is_valid, reason = verify_cryptarithm_solution(mapping, words, result, True)
    assert is_valid, f"Valid solution marked invalid: {reason}"

    # Test case 3: Wrong arithmetic
    mapping = {"A": 1, "B": 2, "C": 3}
    words = ["AB"]  # 12
    result = "AC"  # 13 (wrong!)
    is_valid, reason = verify_cryptarithm_solution(mapping, words, result, True)
    assert not is_valid, "Invalid arithmetic not detected"
    assert "Arithmetic equation not satisfied" in reason

    # Test case 4: Leading zero violation
    mapping = {"A": 0, "B": 1}
    words = ["AB"]  # 01
    result = "AB"  # 01
    is_valid, reason = verify_cryptarithm_solution(mapping, words, result, False)
    assert not is_valid, "Leading zero violation not detected"
    assert "cannot map to 0" in reason

    # Test case 5: Leading zero allowed
    mapping = {"A": 0, "B": 1}
    words = ["AB"]  # 01
    result = "AB"  # 01
    is_valid, reason = verify_cryptarithm_solution(mapping, words, result, True)
    assert is_valid, f"Leading zero incorrectly rejected when allowed: {reason}"

    # Test case 6: Duplicate digit assignments
    mapping = {"A": 1, "B": 1, "C": 2}  # A and B both map to 1
    words = ["AB"]  # Both A and B are in puzzle
    result = "C"  # C is also in puzzle
    is_valid, reason = verify_cryptarithm_solution(mapping, words, result, True)
    assert not is_valid, "Duplicate digits not detected"
    assert "Duplicate digit" in reason

    # Test case 7: Missing letter mapping
    mapping = {"A": 1}  # Missing B
    words = ["AB"]
    result = "AB"
    is_valid, reason = verify_cryptarithm_solution(mapping, words, result, True)
    assert not is_valid, "Missing letter not detected"
    assert "Missing mapping" in reason

    # Test case 8: Extra letter in mapping
    mapping = {"A": 1, "B": 2, "C": 3}  # C is not in puzzle
    words = ["AB"]  # 12
    result = "AB"  # 12
    is_valid, reason = verify_cryptarithm_solution(mapping, words, result, True)
    assert not is_valid, "Extra letter not detected"
    assert "Extra letter" in reason

    # Test case 9: Invalid digit (out of range)
    mapping = {"A": 10, "B": 2}  # 10 is invalid
    words = ["AB"]
    result = "AB"
    is_valid, reason = verify_cryptarithm_solution(mapping, words, result, True)
    assert not is_valid, "Invalid digit not detected"
    assert "Invalid digit" in reason

    # Test case 10: Real cryptarithm example
    # SEND + MORE = MONEY
    # S=9, E=5, N=6, D=7, M=1, O=0, R=8, Y=2
    # 9567 + 1085 = 10652
    mapping = {"S": 9, "E": 5, "N": 6, "D": 7, "M": 1, "O": 0, "R": 8, "Y": 2}
    words = ["SEND", "MORE"]
    result = "MONEY"
    is_valid, reason = verify_cryptarithm_solution(mapping, words, result, False)
    assert is_valid, f"Classic SEND+MORE=MONEY not validated: {reason}"


def test_cryptarithm_curriculum():
    """Test curriculum for cryptarithm dataset"""

    curriculum = CryptarithmCurriculum()
    base_value = {"size": 150, "seed": 1}

    base_cfg: CryptarithmCurriculum = curriculum.generate_configuration(base_value)

    assert base_cfg.seed == 1
    assert base_cfg.size == 150
    assert base_cfg.min_words == 2
    assert base_cfg.max_words == 5

    # Test and validate increase in level
    curriculum.increment_attr_level("words")
    increased_cfg: CryptarithmCurriculum = curriculum.generate_configuration(base_value)

    assert increased_cfg.min_words == 2
    assert increased_cfg.max_words == 10

    # Test and validate decrease in level
    curriculum.decrement_attr_level("words")
    decreased_cfg: CryptarithmCurriculum = curriculum.generate_configuration(base_value)

    assert decreased_cfg.min_words == 2
    assert decreased_cfg.max_words == 5

    # Test upper bound boundary conditions
    for _ in range(10):
        curriculum.increment_attr_level("words")
    upper_bound_cfg: CryptarithmCurriculum = curriculum.generate_configuration(base_value)
    assert upper_bound_cfg.min_words == 2
    assert upper_bound_cfg.max_words == 50

    # Test lower bound boundary conditions
    for _ in range(10):
        curriculum.decrement_attr_level("words")
    lower_bound_cfg: CryptarithmCurriculum = curriculum.generate_configuration(base_value)
    assert lower_bound_cfg.min_words == 2
    assert lower_bound_cfg.max_words == 5
