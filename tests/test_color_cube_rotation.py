import pytest

from reasoning_gym import create_dataset
from reasoning_gym.cognition.color_cube_rotation import Color, ColorCubeRotationDataset, Cube, Side


def test_color_cube_rotation_generation():
    dataset = create_dataset("color_cube_rotation", seed=42, size=10)
    assert isinstance(dataset, ColorCubeRotationDataset)

    for item in dataset:
        # Check required keys exist
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Validate story format
        question = item["question"]
        assert "A cube has:" in question
        assert "side" in question
        assert any(word in question for word in ["rotated", "turned"])
        assert "What is now the color" in question

        # Validate answer is a valid color
        assert item["answer"] in [c.value for c in Color]

        # Validate metadata
        assert "initial_state" in item["metadata"]
        assert "rotations" in item["metadata"]
        assert "target_side" in item["metadata"]
        assert "num_rotations" in item["metadata"]
        assert item["metadata"]["num_rotations"] >= 1


def test_color_cube_rotation_config():
    # Test invalid config raises assertion
    with pytest.raises(AssertionError):
        dataset = create_dataset("color_cube_rotation", min_rotations=0)

    with pytest.raises(AssertionError):
        dataset = create_dataset("color_cube_rotation", max_rotations=1, min_rotations=2)


def test_deterministic_generation():
    dataset1 = create_dataset("color_cube_rotation", seed=42, size=5)
    dataset2 = create_dataset("color_cube_rotation", seed=42, size=5)

    for i in range(5):
        assert dataset1[i]["question"] == dataset2[i]["question"]
        assert dataset1[i]["answer"] == dataset2[i]["answer"]


def test_cube_rotations():
    # Test individual rotation operations
    cube = Cube(
        {
            Side.TOP: Color.RED,
            Side.RIGHT: Color.GREEN,
            Side.FRONT: Color.BLUE,
            Side.LEFT: Color.YELLOW,
            Side.BACK: Color.WHITE,
            Side.BOTTOM: Color.ORANGE,
        }
    )

    # Test front to top rotation
    original = cube.colors.copy()
    cube.rotate_front_to_top()
    assert cube.colors[Side.TOP] == original[Side.FRONT]
    assert cube.colors[Side.FRONT] == original[Side.BOTTOM]
    assert cube.colors[Side.BOTTOM] == original[Side.BACK]
    assert cube.colors[Side.BACK] == original[Side.TOP]
    assert cube.colors[Side.RIGHT] == original[Side.RIGHT]  # Unchanged
    assert cube.colors[Side.LEFT] == original[Side.LEFT]  # Unchanged
