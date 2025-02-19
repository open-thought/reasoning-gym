import pytest

from reasoning_gym.arithmetic import DecimalChainSumConfig, DecimalChainSumDataset


def test_decimal_chain_sum_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = DecimalChainSumConfig(min_terms=0)
        config.validate()

    with pytest.raises(AssertionError):
        config = DecimalChainSumConfig(min_terms=3, max_terms=2)
        config.validate()


def test_decimal_chain_sum_deterministic():
    """Test that dataset generates same items with same seed"""
    config = DecimalChainSumConfig(seed=42, size=10)
    dataset1 = DecimalChainSumDataset(config)
    dataset2 = DecimalChainSumDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_decimal_chain_sum_items():
    """Test basic properties of generated items"""
    config = DecimalChainSumConfig(
        min_terms=2,
        max_terms=4,
        min_digits=1,
        max_digits=2,
        min_decimal_places=1,
        max_decimal_places=2,
        size=100,
        seed=42,
    )
    dataset = DecimalChainSumDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Verify only + and - are used
        expression = item["metadata"]["expression"]
        assert all(op in ["+", "-", " ", "."] or op.isdigit() for op in expression)

        # Check for floating point errors
        numbers = [n for n in expression.split() if any(c.isdigit() for c in n)]
        for num in numbers:
            # Verify no numbers have more decimal places than max_decimal_places
            if "." in num:
                decimal_places = len(num.split(".")[-1])
                assert decimal_places <= config.max_decimal_places, f"Number {num} has more decimal places than allowed"

        # Verify answer has correct precision
        answer_str = item["answer"]
        if "." in answer_str:
            decimal_places = len(answer_str.split(".")[-1])
            assert (
                decimal_places <= config.max_decimal_places
            ), f"Answer {answer_str} has more decimal places than allowed"

        # Verify mathematical correctness within epsilon
        expected = eval(expression)
        assert (
            abs(float(item["answer"]) - expected) < 1e-10
        ), f"Answer {item['answer']} doesn't match expected {expected}"


def test_chain_sum_number_ranges():
    """Test that generated numbers respect digit constraints"""
    # Test 3-digit numbers
    config = DecimalChainSumConfig(
        min_terms=2,
        max_terms=2,  # Fix to 2 terms for easier testing
        min_digits=3,
        max_digits=3,
        min_decimal_places=1,
        max_decimal_places=4,
        size=50,
        seed=42,
    )
    dataset = DecimalChainSumDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        expression = item["metadata"]["expression"]
        numbers = [int(n) for n in expression.split() if n.isdigit()]
        for num in numbers:
            assert 100 <= num <= 999, f"Number {num} outside valid range for 3 digits"

    # Test 1-digit numbers
    config = DecimalChainSumConfig(
        min_terms=2,
        max_terms=2,
        min_digits=1,
        max_digits=1,
        min_decimal_places=1,
        max_decimal_places=4,
        size=50,
        seed=42,
    )
    dataset = DecimalChainSumDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        expression = item["metadata"]["expression"]
        numbers = [int(n) for n in expression.split() if n.isdigit()]
        for num in numbers:
            assert 0 <= num <= 9, f"Number {num} outside valid range for 1 digit"


def test_decimal_chain_sum_negation():
    """Test that negation is properly handled"""
    config = DecimalChainSumConfig(
        min_terms=2,
        max_terms=2,
        min_digits=1,
        max_digits=1,
        min_decimal_places=1,
        max_decimal_places=4,
        allow_negation=True,
        size=50,
        seed=42,
    )
    dataset = DecimalChainSumDataset(config)

    has_positive = False
    has_negative = False

    for i in range(len(dataset)):
        item = dataset[i]
        expression = item["metadata"]["expression"]
        numbers = [float(n) for n in expression.split() if n.replace(".", "").replace("-", "").isdigit()]
        for num in numbers:
            if num > 0:
                has_positive = True
            if num < 0:
                has_negative = True

    assert has_positive and has_negative, "Expected both positive and negative numbers with allow_negation=True"


def test_decimal_chain_sum_iteration():
    """Test that iteration respects dataset size"""
    config = DecimalChainSumConfig(
        min_terms=2,
        max_terms=2,
        min_digits=1,
        max_digits=1,
        min_decimal_places=1,
        max_decimal_places=4,
        size=5,
        seed=42,
    )
    dataset = DecimalChainSumDataset(config)

    items = []
    for item in dataset:
        items.append(item)
    assert len(items) == config.size, "Iterator should yield exactly size items"

    items = list(dataset)
    assert len(items) == config.size, "Iterator should yield exactly size items"

    first_items = list(dataset)
    second_items = list(dataset)
    assert first_items == second_items, "Multiple iterations should yield same items"


def test_decimal_places_generation():
    """Test that generated decimal numbers have correct number of decimal places"""
    # Test fixed decimal places
    config = DecimalChainSumConfig(
        min_terms=2,
        max_terms=2,
        min_digits=1,
        max_digits=2,
        min_decimal_places=2,
        max_decimal_places=2,
        size=50,
        seed=42,
    )
    dataset = DecimalChainSumDataset(config)

    for item in dataset:
        expression = item["metadata"]["expression"]
        # Extract numbers including decimals
        numbers = [n for n in expression.split() if any(c.isdigit() for c in n)]
        for num in numbers:
            decimal_part = num.split(".")[-1]
            assert len(decimal_part) == 2, f"Number {num} should have exactly 2 decimal places"

    # Test varying decimal places
    config = DecimalChainSumConfig(
        min_terms=2,
        max_terms=2,
        min_digits=1,
        max_digits=2,
        min_decimal_places=1,
        max_decimal_places=3,
        size=50,
        seed=42,
    )
    dataset = DecimalChainSumDataset(config)

    for item in dataset:
        expression = item["metadata"]["expression"]
        numbers = [n for n in expression.split() if any(c.isdigit() for c in n)]
        for num in numbers:
            decimal_part = num.split(".")[-1]
            assert 1 <= len(decimal_part) <= 3, f"Number {num} should have between 1 and 3 decimal places"


def test_decimal_precision_scoring():
    """Test that scoring handles decimal precision correctly"""
    config = DecimalChainSumConfig(
        min_terms=2,
        max_terms=2,
        min_digits=1,
        max_digits=2,
        min_decimal_places=2,
        max_decimal_places=3,
        size=1,
        seed=42,
    )
    dataset = DecimalChainSumDataset(config)
    item = dataset[0]

    # Test exact matches with different representations
    assert dataset.score_answer("1.200", {"answer": "1.2"}) == 1.0
    assert dataset.score_answer("1.20", {"answer": "1.200"}) == 1.0
    assert dataset.score_answer("-0.5", {"answer": "-0.500"}) == 1.0

    # Test floating point precision edge cases
    assert dataset.score_answer("0.1", {"answer": "0.100"}) == 1.0
    assert dataset.score_answer("0.3", {"answer": "0.300"}) == 1.0

    # Test incorrect answers
    assert dataset.score_answer("1.200000001", {"answer": "1.200"}) == 0.01
    assert dataset.score_answer("1.199999999", {"answer": "1.200"}) == 0.01

    # Test invalid inputs
    assert dataset.score_answer(None, {"answer": "1.200"}) == 0.0
    assert dataset.score_answer("", {"answer": "1.200"}) == 0.0
    assert dataset.score_answer("invalid", {"answer": "1.200"}) == 0.01
    assert dataset.score_answer("1.2.3", {"answer": "1.200"}) == 0.01
