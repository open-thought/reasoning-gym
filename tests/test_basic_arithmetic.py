import pytest

from reasoning_gym.arithmetic.basic_arithmetic import (
    BasicArithmeticCurriculum,
    BasicArithmeticDataset,
    BasicArithmeticDatasetConfig,
    eval_floordiv,
)
from reasoning_gym.coaching.base_curriculum import DefaultCurriculumContext, RangeAttributeMode


def test_arithmetic_dataset_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = BasicArithmeticDatasetConfig(min_terms=0)
        config.validate()

    with pytest.raises(AssertionError):
        config = BasicArithmeticDatasetConfig(min_terms=3, max_terms=2)
        config.validate()

    with pytest.raises(AssertionError):
        config = BasicArithmeticDatasetConfig(operators=["^"])  # Invalid operator
        config.validate()


def test_arithmetic_dataset_deterministic():
    """Test that dataset generates same items with same seed"""
    config = BasicArithmeticDatasetConfig(seed=42, size=10)
    dataset1 = BasicArithmeticDataset(config)
    dataset2 = BasicArithmeticDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_arithmetic_dataset_items():
    """Test basic properties of generated items"""
    config = BasicArithmeticDatasetConfig(min_terms=2, max_terms=4, min_digits=1, max_digits=2, size=100, seed=42)
    dataset = BasicArithmeticDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Verify the answer matches the expression
        expression = item["metadata"]["expression"]
        answer = eval_floordiv(expression)  # Safe here as we control the expression
        assert str(answer) == item["answer"]


def test_arithmetic_dataset_format_styles():
    """Test different question format styles"""
    config = BasicArithmeticDatasetConfig(
        size=10,
        seed=42,
        format_style="simple",
        min_terms=2,
        max_terms=3,  # Keep expressions simple for testing
        min_digits=1,
        max_digits=2,
    )
    dataset = BasicArithmeticDataset(config)
    assert all(item["question"].strip().endswith(".") for item in dataset)

    config = BasicArithmeticDatasetConfig(
        size=10,
        seed=42,
        format_style="natural",
        min_terms=2,
        max_terms=3,  # Keep expressions simple for testing
        min_digits=1,
        max_digits=2,
    )
    dataset = BasicArithmeticDataset(config)
    assert all(item["question"].strip().endswith(".") or item["question"].strip().endswith("?") for item in dataset)


def test_arithmetic_dataset_iteration():
    """Test that iteration respects dataset size"""
    config = BasicArithmeticDatasetConfig(min_terms=2, max_terms=2, size=5, seed=42)  # Small size for testing
    dataset = BasicArithmeticDataset(config)

    # Test manual iteration
    items = []
    for item in dataset:
        items.append(item)
    assert len(items) == config.size, "Iterator should yield exactly size items"

    # Test list conversion
    items = list(dataset)
    assert len(items) == config.size, "Iterator should yield exactly size items"

    # Test multiple iterations
    first_items = list(dataset)
    second_items = list(dataset)
    assert first_items == second_items, "Multiple iterations should yield same items"


def test_basic_arithmetic_curriculum():
    """Test the BasicArithmeticCurriculum functionality"""
    curriculum = BasicArithmeticCurriculum()

    base_value = {"size": 150, "seed": 1, "min_terms": 2, "max_terms": 2, "min_digits": 1, "max_digits": 1}

    base_cfg: BasicArithmeticDatasetConfig = curriculum.generate_configuration(base_value)
    assert base_cfg.seed == 1
    assert base_cfg.size == 150
    assert base_cfg.min_terms == 2 and base_cfg.max_terms == 2
    assert base_cfg.min_digits == 1 and base_cfg.max_digits == 1

    # Test incrementing attribute levels
    curriculum.increment_attr_level("num_terms")
    curriculum.increment_attr_level("num_digits")
    increased_cfg = curriculum.generate_configuration(base_value)
    assert increased_cfg.min_terms == 2 and increased_cfg.max_terms == 3
    assert increased_cfg.min_digits == 1 and increased_cfg.max_digits == 2

    # Test decrementing attribute level for num_terms
    curriculum.decrement_attr_level("num_terms")
    partially_decreased_cfg = curriculum.generate_configuration(base_value)
    assert partially_decreased_cfg.min_terms == 2 and partially_decreased_cfg.max_terms == 2
    assert partially_decreased_cfg.min_digits == 1 and partially_decreased_cfg.max_digits == 2

    # Test additional increments to ensure levels work as expected
    curriculum.increment_attr_level("num_terms")
    curriculum.increment_attr_level("num_terms")
    higher_level_cfg = curriculum.generate_configuration(base_value)
    assert higher_level_cfg.min_terms == 2 and higher_level_cfg.max_terms == 4
    assert higher_level_cfg.min_digits == 1 and higher_level_cfg.max_digits == 2

    # Test boundary conditions - trying to decrement below level 0
    curriculum.decrement_attr_level("num_terms")
    curriculum.decrement_attr_level("num_terms")
    curriculum.decrement_attr_level("num_digits")
    lower_bound_cfg = curriculum.generate_configuration(base_value)
    assert lower_bound_cfg.min_terms == 2 and lower_bound_cfg.max_terms == 2
    assert lower_bound_cfg.min_digits == 1 and lower_bound_cfg.max_digits == 1

    # Test boundary conditions - trying to increment above max level
    for _ in range(5):
        curriculum.increment_attr_level("num_terms")
        curriculum.increment_attr_level("num_digits")
    upper_bound_cfg = curriculum.generate_configuration(base_value)
    assert upper_bound_cfg.min_terms == 2 and upper_bound_cfg.max_terms == 6
    assert upper_bound_cfg.min_digits == 1 and upper_bound_cfg.max_digits == 4


def test_basic_arithmetic_curriculum_upper_bound():
    curriculum = BasicArithmeticCurriculum()

    base_value = {"size": 150, "seed": 1, "min_terms": 2, "max_terms": 2, "min_digits": 1, "max_digits": 1}

    base_cfg: BasicArithmeticDatasetConfig = curriculum.generate_configuration(
        base_value, context=DefaultCurriculumContext(mode=RangeAttributeMode.UPPER_BOUND)
    )
    assert base_cfg.seed == 1
    assert base_cfg.size == 150
    assert base_cfg.min_terms == 2 and base_cfg.max_terms == 2
    assert base_cfg.min_digits == 1 and base_cfg.max_digits == 1

    # Test incrementing attribute levels
    curriculum.increment_attr_level("num_terms")
    curriculum.increment_attr_level("num_digits")
    increased_cfg = curriculum.generate_configuration(base_value)
    assert increased_cfg.min_terms == 2 and increased_cfg.max_terms == 3
    assert increased_cfg.min_digits == 1 and increased_cfg.max_digits == 2


def test_arithmetic_dataset_large_random_generation():
    """Stress-test generation of many arithmetic questions to catch random errors"""
    config = BasicArithmeticDatasetConfig(
        size=100000,
        seed=123,
        min_terms=2,
        max_terms=6,
        min_digits=1,
        max_digits=3,
    )
    dataset = BasicArithmeticDataset(config)
    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item and "answer" in item and "metadata" in item
