from magiccube.cube import Cube
from reasoning_gym.graphs.quantum_lock import QuantumLockConfig, QuantumLockDataset


def test_quantumlock_items():
    """Test basic properties and solution of generated items"""
    config = QuantumLockConfig(
        difficulty=10,
    )
    dataset = QuantumLockDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata contains required fields
        assert "solution_path" in item["metadata"]
        assert "difficulty" in item["metadata"]

        assert dataset.score_answer(answer=item['metadata']['solution_path'], entry=item) == 1.0
        assert dataset.score_answer(answer=None, entry=item) == 0.0
