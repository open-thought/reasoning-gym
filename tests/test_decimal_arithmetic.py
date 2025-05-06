import pytest

from reasoning_gym.arithmetic.decimal_arithmetic import (
    DecimalArithmeticConfig,
    DecimalArithmeticCurriculum,
    DecimalArithmeticDataset,
)


def test_decimal_arithmetic():
    """Test basic properties and solution of generated items"""

    # Easy
    config = DecimalArithmeticConfig(
        seed=42, size=2000, min_num_decimal_places=3, max_num_decimal_places=3, precision=5, min_terms=2, max_terms=3
    )
    dataset = DecimalArithmeticDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Test the scoring
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0

    # M
    config = DecimalArithmeticConfig(
        seed=42, size=2000, min_num_decimal_places=3, max_num_decimal_places=6, precision=8, min_terms=3, max_terms=5
    )
    dataset = DecimalArithmeticDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0

    # H
    config = DecimalArithmeticConfig(
        seed=42, size=2000, min_num_decimal_places=3, max_num_decimal_places=13, precision=15, min_terms=3, max_terms=5
    )
    dataset = DecimalArithmeticDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0


def test_decimal_arithmetic_curriculum():
    """Test the decimal arithmetic curriculum generation and attribute adjustment"""
    curriculum = DecimalArithmeticCurriculum()

    base_value = {"size": 200, "seed": 42, "precision": 6}

    base_cfg: DecimalArithmeticConfig = curriculum.generate_configuration(base_value)
    assert base_cfg.seed == 42
    assert base_cfg.size == 200
    assert base_cfg.precision == 6
    assert base_cfg.min_num_decimal_places == 2 and base_cfg.max_num_decimal_places == 4

    # Test incrementing attribute level
    curriculum.increment_attr_level("decimal_places")
    increased_cfg = curriculum.generate_configuration(base_value)
    assert increased_cfg.min_num_decimal_places == 2 and increased_cfg.max_num_decimal_places == 6

    # Test incrementing attribute level again
    curriculum.increment_attr_level("decimal_places")
    further_increased_cfg = curriculum.generate_configuration(base_value)
    assert further_increased_cfg.min_num_decimal_places == 2 and further_increased_cfg.max_num_decimal_places == 8

    # Test decrementing attribute level
    curriculum.decrement_attr_level("decimal_places")
    decreased_cfg = curriculum.generate_configuration(base_value)
    assert decreased_cfg.min_num_decimal_places == 2 and decreased_cfg.max_num_decimal_places == 6

    # Test decrementing attribute level to base level
    curriculum.decrement_attr_level("decimal_places")
    base_level_cfg = curriculum.generate_configuration(base_value)
    assert base_level_cfg.min_num_decimal_places == 2 and base_level_cfg.max_num_decimal_places == 4
