import pytest

from reasoning_gym.cognition.rubiks_cube import RubiksCubeConfig, RubiksCubeDataset


def test_rubikscube_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = RubiksCubeConfig(cube_size=1)  # Too small
        config.validate()

    with pytest.raises(AssertionError):
        config = RubiksCubeConfig(scramble_steps=0)  # Don't give an unscrambled cube
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
        scramble_steps=4,
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
            assert dataset.score_answer(answer="R", entry=item) == 0.05

        if len(item["metadata"]["example_correct_answer"]) > 0:
            assert dataset.score_answer(answer="", entry=item) == 0.01
