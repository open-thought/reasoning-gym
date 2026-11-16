import pytest

from reasoning_gym.cognition.pyraminx import (
    PyraminxConfig,
    PyraminxCurriculum,
    PyraminxDataset,
    PyraminxState,
    solve_pyraminx,
)


def test_pyraminx_config_validation():
    """Test that invalid configurations raise appropriate errors."""
    with pytest.raises(AssertionError):
        config = PyraminxConfig(min_scramble_steps=0)
        config.validate()

    with pytest.raises(AssertionError):
        config = PyraminxConfig(min_scramble_steps=5, max_scramble_steps=2)
        config.validate()

    with pytest.raises(AssertionError):
        config = PyraminxConfig(size=0)
        config.validate()


def test_pyraminx_state_and_moves():
    """Test puzzle state moves, inverses, and properties."""
    state = PyraminxState()
    assert state.is_solved()
    assert state.partial_score() == 1.0

    # Test M^3 == Identity for all moves
    for m in ["U", "L", "R", "B", "u", "l", "r", "b"]:
        st = state
        for _ in range(3):
            st = st.apply_move(m)
        assert st.is_solved()

    # Test M * M' == Identity
    for m in ["U", "L", "R", "B", "u", "l", "r", "b"]:
        st = state.apply_move(m)
        st = st.apply_move(f"{m}'")
        assert st.is_solved()


def test_pyraminx_deterministic():
    """Test that dataset generates identical items with same seed."""
    config = PyraminxConfig(seed=42, size=10)
    dataset1 = PyraminxDataset(config)
    dataset2 = PyraminxDataset(config)

    assert len(dataset1) == 10
    assert len(dataset2) == 10

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_pyraminx_items():
    """Test dataset item generation and scoring."""
    config = PyraminxConfig(
        min_scramble_steps=2,
        max_scramble_steps=4,
        include_tips=True,
        seed=123,
        size=20,
    )
    dataset = PyraminxDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        meta = item["metadata"]
        assert meta["source_dataset"] == "pyraminx"
        assert "scramble_steps" in meta
        assert "scramble_moves" in meta
        assert "example_correct_answer" in meta

        # Correct answer must receive perfect score 1.0
        score = dataset.score_answer(meta["example_correct_answer"], item)
        assert score == 1.0, f"Expected 1.0, got {score} for {meta['example_correct_answer']}"

        # Wrong answers
        assert dataset.score_answer("wrong moves", item) <= 0.5
        assert dataset.score_answer(None, item) == 0.0
        assert dataset.score_answer("", item) == 0.0


def test_pyraminx_curriculum():
    """Test curriculum progression levels."""
    curriculum = PyraminxCurriculum()
    base_value = {"size": 100, "seed": 7}

    cfg: PyraminxConfig = curriculum.generate_configuration(base_value)
    assert cfg.min_scramble_steps == 2
    assert cfg.max_scramble_steps == 4
    assert not cfg.include_tips

    # Increment scramble_steps
    curriculum.increment_attr_level("scramble_steps")
    cfg2 = curriculum.generate_configuration(base_value)
    assert cfg2.min_scramble_steps == 2
    assert cfg2.max_scramble_steps == 6

    # Increment include_tips
    curriculum.increment_attr_level("include_tips")
    cfg3 = curriculum.generate_configuration(base_value)
    assert cfg3.include_tips is True

    # Decrement
    curriculum.decrement_attr_level("scramble_steps")
    cfg4 = curriculum.generate_configuration(base_value)
    assert cfg4.max_scramble_steps == 4
