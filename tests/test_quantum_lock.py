import pytest

from reasoning_gym.graphs.quantum_lock import QuantumLockConfig, QuantumLockDataset


def test_quantumlock_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = QuantumLockConfig(difficulty=-1)
        config.validate()

    with pytest.raises(AssertionError):
        config = QuantumLockConfig(size=0)
        config.validate()


def test_quantumlock_deterministic():
    """Test that dataset generates same items with same seed"""
    config = QuantumLockConfig(seed=42, size=10)
    dataset1 = QuantumLockDataset(config)
    dataset2 = QuantumLockDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_quantumlock_items():
    """Test basic properties and solution of generated items"""
    config = QuantumLockConfig(difficulty=10, size=25)
    dataset = QuantumLockDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata contains required fields
        assert "solution_path" in item["metadata"]
        assert "difficulty" in item["metadata"]
        assert "buttons" in item["metadata"]
        assert "initial_state" in item["metadata"]
        assert "target_value" in item["metadata"]

        # Verify solution works
        assert dataset.score_answer(answer=item["metadata"]["solution_path"], entry=item) == 1.0
        assert dataset.score_answer(answer=None, entry=item) == 0.0


def test_quantumlock_button_states():
    """Test button state transitions and validity"""
    config = QuantumLockConfig(difficulty=5, size=10)
    dataset = QuantumLockDataset(config)

    for item in dataset:
        buttons = item["metadata"]["buttons"]

        # Check button properties
        for btn in buttons:
            assert "name" in btn
            assert "type" in btn
            assert "value" in btn
            assert "active_state" in btn

            # Verify button name format
            assert btn["name"] in ["A", "B", "C"]

            # Verify operation type
            assert btn["type"] in ["add", "subtract", "multiply"]

            # Verify state constraints
            assert btn["active_state"] in ["red", "green", "any"]


def test_quantumlock_solution_validation():
    """Test solution validation and simulation"""
    config = QuantumLockConfig(difficulty=5, size=10)
    dataset = QuantumLockDataset(config)

    for item in dataset:
        solution = item["metadata"]["solution_path"]
        target = item["metadata"]["target_value"]

        # Test solution simulation
        final_value = dataset.simulate_sequence(item["metadata"], solution)
        assert final_value == target

        # Test invalid button sequences
        assert (
            dataset.simulate_sequence(item["metadata"], ["X", "Y", "Z"])  # Invalid buttons
            == item["metadata"]["initial_value"]
        )


def test_quantumlock_scoring():
    """Test score calculation for various answers"""
    config = QuantumLockConfig(difficulty=5, size=10)
    dataset = QuantumLockDataset(config)

    for item in dataset:
        solution = item["metadata"]["solution_path"]

        # Test correct solution
        assert dataset.score_answer(solution, item) == 1.0

        # Test empty/None answers
        assert dataset.score_answer(None, item) == 0.0
        assert dataset.score_answer("", item) == 0.1

        # Test invalid buttons
        assert dataset.score_answer("XYZ", item) == 0.1

        # Test case insensitivity
        if solution:
            lower_solution = "".join(solution).lower()
            assert dataset.score_answer(lower_solution, item) == 1.0
