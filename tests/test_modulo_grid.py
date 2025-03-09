import pytest

from reasoning_gym.cognition.modulo_grid import ModuloGridConfig, ModuloGridCurriculum, ModuloGridDataset


def test_modulo_grid():
    """Test basic properties and solution of generated items"""

    # Easy
    config = ModuloGridConfig(seed=42, size=50, size_x=10, size_y=10, max_divisor=10, max_target=10, max_holes=1)
    dataset = ModuloGridDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item and isinstance(item["question"], str)
        assert "answer" in item and isinstance(item["answer"], str)
        assert "metadata" in item

        # Test the scoring
        assert item["question"] != item["answer"]
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0
        assert dataset.score_answer(answer=None, entry=item) == 0.0

    # Hard
    config = ModuloGridConfig(seed=42, size=50, size_x=25, size_y=25, max_divisor=25, max_target=25, max_holes=15)
    dataset = ModuloGridDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item and isinstance(item["question"], str)
        assert "answer" in item and isinstance(item["answer"], str)
        assert "metadata" in item

        # Test the scoring
        assert item["question"] != item["answer"]
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0
        assert dataset.score_answer(answer=None, entry=item) == 0.0


def test_mg_curriculum():
    curriculum = ModuloGridCurriculum()

    base_value = {"size": 150, "seed": 1}

    base_cfg: ModuloGridConfig = curriculum.generate_configuration(base_value)
    assert base_cfg.seed == 1
    assert base_cfg.size == 150
    assert base_cfg.size_x == 20
    curriculum.increment_attr_level("size_x")
    increased_cfg = curriculum.generate_configuration(base_value)
    assert increased_cfg.size_x == 30
