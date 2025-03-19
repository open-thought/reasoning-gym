import pytest

from reasoning_gym.arc.board_format import format_board
from reasoning_gym.arc.rearc import ReArcConfig, ReArcCurriculum, ReArcDataset


def test_rearc_config_validation():
    """Test validation of ReArc configuration parameters"""
    with pytest.raises(AssertionError):
        ReArcConfig(diff_lb=0.5, diff_ub=0.3).validate()

    with pytest.raises(AssertionError):
        ReArcConfig(size=0).validate()


def test_rearc_deterministic():
    """Test dataset reproducibility with fixed seed"""
    config = ReArcConfig(seed=42, size=100, diff_lb=0, diff_ub=1)
    ds1 = ReArcDataset(config)
    ds2 = ReArcDataset(config)

    for i in range(len(ds1)):
        assert ds1[i] == ds2[i], "ReArc datasets with same seed should match exactly"


def test_rearc_items():
    """Test basic structure and metadata of generated items"""
    config = ReArcConfig(seed=42, size=100, diff_lb=0, diff_ub=1)
    dataset = ReArcDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        meta = item["metadata"]
        assert "input" in meta
        assert "output" in meta
        assert "task_id" in meta
        assert "rng_difficulty" in meta["difficulty"]
        assert "pso_difficulty" in meta["difficulty"]

        # Validate difficulty bounds
        assert config.diff_lb <= meta["difficulty"]["rng_difficulty"] <= config.diff_ub
        assert config.diff_lb <= meta["difficulty"]["pso_difficulty"] <= config.diff_ub


def test_rearc_solution_validation():
    """Test solution verification and scoring"""
    config = ReArcConfig(size=100, seed=123)
    dataset = ReArcDataset(config)

    for item in dataset:
        # Test correct solution
        correct = format_board(item["metadata"]["output"], dataset.board_format_opts)
        assert dataset.score_answer(correct, entry=item) == 1.0

        # Test invalid format
        invalid_grid = """
9 9 9
1 2 1
7 8 7
0 0 0
"""
        assert dataset.score_answer(invalid_grid, entry=item) == 0.05

        # Test empty answer
        assert dataset.score_answer(None, entry=item) == 0.0


def test_rearc_scoring_edge_cases():
    """Test scoring for partial and malformed answers"""
    config = ReArcConfig(size=100, seed=456)
    dataset = ReArcDataset(config)

    for item in dataset:
        # Partial match
        partial = format_board([[0, 0], [0, 0]], dataset.board_format_opts)
        assert 0.0 < dataset.score_answer(partial, entry=item) < 1.0

        # Malformed answer
        assert dataset.score_answer("[[invalid", entry=item) == 0.0

        # Case sensitivity
        answer = format_board(item["metadata"]["output"], dataset.board_format_opts).lower()
        assert dataset.score_answer(answer, entry=item) == 1.0


def test_rearc_curriculum():
    """Test the ReArc curriculum functionality"""
    curriculum = ReArcCurriculum()

    base_value = {"size": 50, "seed": 42}

    # Test default configuration
    base_cfg: ReArcConfig = curriculum.generate_configuration(base_value)
    assert base_cfg.seed == 42
    assert base_cfg.size == 50

    # Default levels should have weights that select only the easiest tasks
    assert base_cfg.pso_difficulty_weights == [1, 0, 0, 0, 0, 0, 0, 0]
    assert base_cfg.rng_difficulty_weights == [1, 0, 0, 0, 0, 0, 0, 0]

    # Test incrementing pso_difficulty attribute
    curriculum.increment_attr_level("pso_difficulty")
    pso_cfg = curriculum.generate_configuration(base_value)
    assert pso_cfg.pso_difficulty_weights == [0, 1, 0, 0, 0, 0, 0, 0]  # Level 1: second difficulty range
    assert pso_cfg.rng_difficulty_weights == [1, 0, 0, 0, 0, 0, 0, 0]  # RNG unchanged

    # Test incrementing rng_difficulty attribute
    curriculum.increment_attr_level("rng_difficulty")
    rng_cfg = curriculum.generate_configuration(base_value)
    assert rng_cfg.pso_difficulty_weights == [0, 1, 0, 0, 0, 0, 0, 0]  # PSO unchanged
    assert rng_cfg.rng_difficulty_weights == [0, 1, 0, 0, 0, 0, 0, 0]  # Level 1: second difficulty range

    # Test decrementing pso_difficulty attribute
    curriculum.decrement_attr_level("pso_difficulty")
    decr_cfg = curriculum.generate_configuration(base_value)
    assert decr_cfg.pso_difficulty_weights == [1, 0, 0, 0, 0, 0, 0, 0]  # Back to level 0
    assert decr_cfg.rng_difficulty_weights == [0, 1, 0, 0, 0, 0, 0, 0]  # RNG unchanged

    # Test global level setting to higher level
    curriculum.set_global_level(3)  # Set all attributes to level 3
    global_cfg = curriculum.generate_configuration(base_value)
    assert global_cfg.pso_difficulty_weights == [0, 0, 0, 1, 0, 0, 0, 0]  # Level 3
    assert global_cfg.rng_difficulty_weights == [0, 0, 0, 1, 0, 0, 0, 0]  # Level 3

    # Test increment global level
    curriculum.increment_global_level()  # Should go to level 4
    incr_global_cfg = curriculum.generate_configuration(base_value)
    assert incr_global_cfg.pso_difficulty_weights == [0, 0, 0, 0, 1, 0, 0, 0]  # Level 4
    assert incr_global_cfg.rng_difficulty_weights == [0, 0, 0, 0, 1, 0, 0, 0]  # Level 4

    # Test decrement global level
    curriculum.decrement_global_level()  # Should go back to level 3
    decr_global_cfg = curriculum.generate_configuration(base_value)
    assert decr_global_cfg.pso_difficulty_weights == [0, 0, 0, 1, 0, 0, 0, 0]  # Level 3
    assert decr_global_cfg.rng_difficulty_weights == [0, 0, 0, 1, 0, 0, 0, 0]  # Level 3
