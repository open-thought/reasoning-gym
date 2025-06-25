import pytest

from reasoning_gym.games.countdown import CountdownConfig, CountdownCurriculum, CountdownDataset


def test_countdown_game_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = CountdownConfig(min_numbers=1)  # Too few numbers
        config.validate()

    with pytest.raises(AssertionError):
        config = CountdownConfig(min_numbers=4, max_numbers=3)  # Invalid range
        config.validate()

    with pytest.raises(AssertionError):
        config = CountdownConfig(operators=["^"])  # Invalid operator
        config.validate()


def test_countdown_game_deterministic():
    """Test that dataset generates same items with same seed"""
    config = CountdownConfig(seed=42, size=10)
    dataset1 = CountdownDataset(config)
    dataset2 = CountdownDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_countdown_game_items():
    """Test basic properties of generated items"""
    config = CountdownConfig(
        min_numbers=3,
        max_numbers=5,
        min_value=1,
        max_value=20,  # Small numbers for testing
        min_target=10,
        max_target=100,
        size=50,
        seed=42,
    )
    dataset = CountdownDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata contains required fields
        assert "numbers" in item["metadata"]
        assert "target" in item["metadata"]
        assert "expression" in item["metadata"]

        # Verify number of source numbers is within config range
        assert config.min_numbers <= len(item["metadata"]["numbers"]) <= config.max_numbers

        # Verify target is within config range
        assert config.min_target <= item["metadata"]["target"] <= config.max_target

        # Verify all numbers are within config range
        assert all(config.min_value <= n <= config.max_value for n in item["metadata"]["numbers"])

        # Verify expression evaluates correctly
        expr = item["metadata"]["expression"]

        # check score
        assert dataset.score_answer(answer=expr, entry=item) == 1.0  # correct answer
        assert dataset.score_answer(answer="45+2", entry=item) == 0.05  # wrong answer but an attempt
        assert (
            dataset.score_answer(answer="a wrong solution", entry=item) == 0.01
        )  # wrong answer but incorrectly formatted
        assert dataset.score_answer(answer="", entry=item) == 0.01  # wrong answer but empty string
        assert dataset.score_answer(answer=None, entry=item) == 0.01  # no answer

        try:
            result = eval(expr)  # Safe here since we control expression generation
            assert result == item["metadata"]["target"]
        except (SyntaxError, ZeroDivisionError):
            pytest.fail(f"Invalid expression generated: {expr}")


def test_answer_with_incorrect_numbers():
    dataset = CountdownDataset(CountdownConfig(size=10, seed=42))
    answer = "45+2"
    item = {
        "metadata": {
            "numbers": [44, 3],
            "target": 47,
        }
    }
    assert dataset.score_answer(answer=answer, entry=item) == 0.05


def test_answer_without_all_numbers():
    dataset = CountdownDataset(CountdownConfig(size=10, seed=42))
    answer = "45+2+3"
    item = {"metadata": {"numbers": [1, 45, 2, 3], "target": 50}}
    assert dataset.score_answer(answer=answer, entry=item) == 0.05


def test_edge_cases_1():
    dataset = CountdownDataset(CountdownConfig(size=10, seed=42))
    answer = "1*81"
    item = {"metadata": {"numbers": [1, 1, 1, 81], "target": 81}}
    assert dataset.score_answer(answer=answer, entry=item) != 1.0


def test_edge_cases_2():
    dataset = CountdownDataset(CountdownConfig(size=10, seed=42))
    answer = "6*34/11-1"
    item = {"metadata": {"numbers": [6, 34, 1, 11], "target": 17}}
    assert dataset.score_answer(answer=answer, entry=item) != 1.0


def test_countdown_game_randomization():
    """Test number randomization configuration"""
    config = CountdownConfig(min_numbers=4, max_numbers=4, shuffle=False, size=10, seed=42)  # Fixed size for testing

    # Without randomization, numbers should appear in same order
    dataset = CountdownDataset(config)
    first_item = dataset[0]
    expr_nums = [
        int(n) for n in first_item["metadata"]["expression"].replace("(", "").replace(")", "").split(" ") if n.isdigit()
    ]
    assert expr_nums == first_item["metadata"]["numbers"]

    # With randomization, numbers might appear in different order
    config.shuffle = True
    dataset = CountdownDataset(config)
    first_item = dataset[0]
    expr_nums = [
        int(n) for n in first_item["metadata"]["expression"].replace("(", "").replace(")", "").split(" ") if n.isdigit()
    ]
    assert sorted(expr_nums) == sorted(first_item["metadata"]["numbers"])


def test_countdown_curriculum():
    curriculum = CountdownCurriculum()

    base_value = {"size": 150, "seed": 1}

    print(base_value)

    base_cfg: CountdownConfig = curriculum.generate_configuration(base_value)
    assert base_cfg.seed == 1
    assert base_cfg.size == 150
    assert base_cfg.min_numbers == 3 and base_cfg.max_numbers == 6
    assert base_cfg.min_target == 100 and base_cfg.max_target == 500
    assert base_cfg.min_value == 1 and base_cfg.max_value == 100

    # Test incrementing attribute levels
    curriculum.increment_attr_level("numbers")
    curriculum.increment_attr_level("target")
    curriculum.increment_attr_level("value")
    increased_cfg = curriculum.generate_configuration(base_value)
    assert increased_cfg.min_numbers == 3 and increased_cfg.max_numbers == 9
    assert increased_cfg.min_target == 100 and increased_cfg.max_target == 1000
    assert increased_cfg.min_value == 1 and increased_cfg.max_value == 200

    # Test decrementing attribute level for numbers again
    curriculum.decrement_attr_level("numbers")
    partially_decreased_cfg = curriculum.generate_configuration(base_value)
    assert partially_decreased_cfg.min_numbers == 3 and partially_decreased_cfg.max_numbers == 6
    assert partially_decreased_cfg.min_target == 100 and partially_decreased_cfg.max_target == 1000
    assert partially_decreased_cfg.min_value == 1 and partially_decreased_cfg.max_value == 200
