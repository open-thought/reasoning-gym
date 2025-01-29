import pytest

from magiccube.cube import Cube
from reasoning_gym.cognition.rubiks_cube import RubiksCubeConfig, RubiksCubeDataset


def test_rubikscube_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = RubiksCubeConfig(cube_size=1)  # Too small
        config.validate()

    with pytest.raises(AssertionError):
        config = RubiksCubeConfig(scramble_steps=0)  # Don't give an unscrambled cube
        config.validate()


def test_rubikscube_items():
    """Test basic properties and solution of generated items"""
    config = RubiksCubeConfig(
        cube_size=3,
        scramble_steps=4
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

        assert dataset.score_answer(answer=item['metadata']['example_correct_answer'], entry=item) == 1.0
        assert dataset.score_answer(answer='R', entry=item) == 0.01
        assert dataset.score_answer(answer=None, entry=item) == 0.0

