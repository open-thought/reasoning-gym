import pytest

from reasoning_gym.arc.board_format import format_board
from reasoning_gym.arc.rearc import ReArcConfig, ReArcDataset


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
        assert "rng" in meta["difficulty"]
        assert "pso" in meta["difficulty"]

        # Validate difficulty bounds
        assert config.diff_lb <= meta["difficulty"]["rng"] <= config.diff_ub
        assert config.diff_lb <= meta["difficulty"]["pso"] <= config.diff_ub


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
        assert dataset.score_answer("[[invalid", entry=item) == 0.01

        # Case sensitivity
        answer = format_board(item["metadata"]["output"], dataset.board_format_opts).lower()
        assert dataset.score_answer(answer, entry=item) == 1.0
