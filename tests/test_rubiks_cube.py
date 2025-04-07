import pytest

from reasoning_gym.cognition.rubiks_cube import RubiksCubeConfig, RubiksCubeCurriculum, RubiksCubeDataset


def test_rubikscube_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = RubiksCubeConfig(cube_size=1)  # Too small
        config.validate()

    with pytest.raises(AssertionError):
        config = RubiksCubeConfig(max_scramble_steps=0)  # Don't give an unscrambled cube
        config.validate()


def test_rubikscube_deterministic():
    """Test that dataset generates same items with same seed"""
    config = RubiksCubeConfig(seed=42, size=15)  # Only check first 15 entries for speed
    dataset1 = RubiksCubeDataset(config)
    dataset2 = RubiksCubeDataset(config)
    assert len(dataset1) == 15
    assert len(dataset2) == 15

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_rubikscube_items():
    """Test basic properties and solution of generated items"""
    config = RubiksCubeConfig(
        cube_size=3,
        min_scramble_steps=4,
        max_scramble_steps=4,
        seed=42,
        size=100,
    )
    dataset = RubiksCubeDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata contains required fields
        assert "cube_size" in item["metadata"]
        assert "cube_size" in item["metadata"]
        assert "scramble_steps" in item["metadata"]
        assert "scramble_moves" in item["metadata"]
        assert "example_correct_answer" in item["metadata"]

        assert dataset.score_answer(answer=item["metadata"]["example_correct_answer"], entry=item) == 1.0
        assert dataset.score_answer(answer="a wrong solution", entry=item) == 0.01
        assert dataset.score_answer(answer=None, entry=item) == 0.0

        if item["metadata"]["example_correct_answer"] != "R":
            assert dataset.score_answer(answer="R", entry=item) == 0.01

        assert dataset.score_answer(answer="R2 R3 R4 R5 R'2 R'3", entry=item) == 0.01

        if len(item["metadata"]["example_correct_answer"]) > 0:
            assert dataset.score_answer(answer="", entry=item) == 0.01


def test_rubiks_cube_curriculum():
    curriculum = RubiksCubeCurriculum()

    base_value = {"size": 150, "seed": 1}

    base_cfg: RubiksCubeConfig = curriculum.generate_configuration(base_value)
    assert base_cfg.seed == 1
    assert base_cfg.size == 150
    assert base_cfg.cube_size == 3
    assert base_cfg.min_scramble_steps == 3 and base_cfg.max_scramble_steps == 10

    # test incrementing attribute levels for cube_size & scramble_stepsd attributes
    curriculum.increment_attr_level("cube_size")
    curriculum.increment_attr_level("scramble_steps")
    increased_cfg = curriculum.generate_configuration(base_value)
    assert increased_cfg.cube_size == 4
    assert increased_cfg.min_scramble_steps == 3 and increased_cfg.max_scramble_steps == 25

    # test decrementing attribute level for cube_size again
    curriculum.decrement_attr_level("cube_size")
    partially_decreased_cfg = curriculum.generate_configuration(base_value)
    assert partially_decreased_cfg.cube_size == 3
    assert partially_decreased_cfg.min_scramble_steps == 3 and partially_decreased_cfg.max_scramble_steps == 25
