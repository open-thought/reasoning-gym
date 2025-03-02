import pytest

from reasoning_gym.utils import compute_decimal_reward, extract_answer, format_number


def test_extract_answer():
    assert extract_answer("This is a text. <final_answer>1234</final_answer>", tag_name="final_answer") == "1234"

    # ignore whitespaces
    assert extract_answer("This is a text. <answer>\n1234 </answer>", tag_name="answer", strip=True) == "1234"


def test_format_number():
    # Test integers
    assert format_number(42) == "42"
    assert format_number(42.0) == "42"

    # Test decimals
    assert format_number(3.14) == "3.14"
    assert format_number(3.10) == "3.1"
    assert format_number(3.00) == "3"

    # Test with max_decimals (rounding)
    assert format_number(3.14159, max_decimals=4, round_if_needed=True) == "3.1416"

    # Test with trailing zeros
    assert format_number(5.5000) == "5.5"

    # Test error cases
    with pytest.raises(ValueError):
        format_number(3.14159, max_decimals=2)


def test_compute_decimal_reward():
    # Test exact matches
    assert compute_decimal_reward("42", "42") == 1.0
    assert compute_decimal_reward("3.14", "3.14") == 1.0

    # Test with commas
    assert compute_decimal_reward("1,000", "1000") == 1.0
    assert compute_decimal_reward("1,000", "1000", strip_commas=False) < 1.0

    # Test partial matches
    assert compute_decimal_reward("The answer is 42", "42") < 1.0
    assert compute_decimal_reward("The answer is 42", "42") > 0.01

    # Test invalid answers
    assert compute_decimal_reward(None, "42") == 0.0
    assert compute_decimal_reward("", "42") == 0.0
    assert compute_decimal_reward("not a number", "42") == 0.01
