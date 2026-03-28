"""Tests for Path Star graph problem generation"""

import pytest

from reasoning_gym.graphs.path_star import PathStarConfig, PathStarCurriculum, PathStarDataset


def test_path_star_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = PathStarConfig(degree=1)  # Must be >= 2
        config.validate()

    with pytest.raises(AssertionError):
        config = PathStarConfig(min_path_length=1)  # Must be >= 2
        config.validate()

    with pytest.raises(AssertionError):
        config = PathStarConfig(min_path_length=5, max_path_length=3)  # min > max
        config.validate()

    with pytest.raises(AssertionError):
        config = PathStarConfig(degree=3, max_path_length=5, node_range=16)  # node_range too small (need > 3*5+1=16)
        config.validate()


def test_path_star_dataset_deterministic():
    """Test that dataset generates same items with same seed"""
    config = PathStarConfig(seed=42, size=10)
    dataset1 = PathStarDataset(config)
    dataset2 = PathStarDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_path_star_dataset_items():
    """Test basic properties of generated items"""
    config = PathStarConfig(min_path_length=3, max_path_length=5, size=10, seed=42)
    dataset = PathStarDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        # Check item structure
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata fields
        assert item["metadata"]["source_dataset"] == "path_star"
        assert item["metadata"]["source_index"] == i
        assert "center" in item["metadata"]
        assert "goal" in item["metadata"]
        assert "path_length" in item["metadata"]
        assert "goal_path" in item["metadata"]
        assert "difficulty" in item["metadata"]

        # Verify answer format: space-separated integers
        answer_parts = item["answer"].split()
        assert all(part.isdigit() for part in answer_parts)

        # First node should be center, last should be goal
        center = item["metadata"]["center"]
        goal = item["metadata"]["goal"]
        assert int(answer_parts[0]) == center
        assert int(answer_parts[-1]) == goal

        # Path length should match: center + path_length nodes
        path_length = item["metadata"]["path_length"]
        assert len(answer_parts) == path_length + 1

        # Path length within configured range
        assert config.min_path_length <= path_length <= config.max_path_length


def test_path_star_dataset_iteration():
    """Test that iteration respects dataset size"""
    config = PathStarConfig(size=5, seed=42)
    dataset = PathStarDataset(config)

    items = list(dataset)
    assert len(items) == config.size

    # Test multiple iterations yield same items
    assert items == list(dataset)


def test_path_star_answer_correctness():
    """Test that generated paths are valid by checking edge connectivity"""
    config = PathStarConfig(size=20, seed=123)
    dataset = PathStarDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        question = item["question"]
        answer_parts = [int(x) for x in item["answer"].split()]

        # Parse edges from the question
        # Format: ...edges_str/start goal = ...
        # Extract the task part between "Solve the following task:\n" and the end
        task_line = question.split("Solve the following task:\n")[1].strip()
        edge_part, _ = task_line.split("/")
        edges = set()
        for edge_str in edge_part.split("|"):
            edge_str = edge_str.strip()
            if edge_str:
                u, v = edge_str.split()
                edges.add((int(u), int(v)))

        # Verify consecutive nodes in the answer are connected by edges
        for j in range(len(answer_parts) - 1):
            u, v = answer_parts[j], answer_parts[j + 1]
            assert (u, v) in edges, f"Edge ({u}, {v}) not found in edges for item {i}"


def test_path_star_score_answer():
    """Test the score_answer method"""
    config = PathStarConfig(seed=42, size=5)
    dataset = PathStarDataset(config)
    item = dataset[0]
    oracle = item["answer"]

    # Exact match
    assert dataset.score_answer(oracle, item) == 1.0

    # Match with extra whitespace
    assert dataset.score_answer(f"  {oracle}  ", item) == 1.0

    # Match with extra internal whitespace
    spaced = oracle.replace(" ", "  ")
    assert dataset.score_answer(spaced, item) == 1.0

    # Wrong answer
    assert dataset.score_answer("0 1 2 3", item) == 0.0

    # None
    assert dataset.score_answer(None, item) == 0.0

    # Empty string
    assert dataset.score_answer("", item) == 0.0


def test_path_star_reversed():
    """Test that reversed=True produces correct answer and task format"""
    config_fwd = PathStarConfig(seed=42, size=5, reversed=False)
    config_rev = PathStarConfig(seed=42, size=5, reversed=True)
    dataset_fwd = PathStarDataset(config_fwd)
    dataset_rev = PathStarDataset(config_rev)

    for i in range(len(dataset_fwd)):
        item_fwd = dataset_fwd[i]
        item_rev = dataset_rev[i]

        # Reversed answer should be the forward answer reversed
        fwd_parts = item_fwd["answer"].split()
        rev_parts = item_rev["answer"].split()
        assert rev_parts == list(reversed(fwd_parts))

        # Task format should swap start/goal
        center = item_fwd["metadata"]["center"]
        goal = item_fwd["metadata"]["goal"]
        assert f"/{center} {goal} = " in item_fwd["question"]
        assert f"/{goal} {center} = " in item_rev["question"]


def test_path_star_curriculum():
    """Test curriculum creates valid configs at various levels"""
    curriculum = PathStarCurriculum()

    base_value = {"size": 150, "seed": 1}

    # Level 0 (base)
    base_cfg: PathStarConfig = curriculum.generate_configuration(base_value)
    assert base_cfg.seed == 1
    assert base_cfg.size == 150
    assert base_cfg.degree == 2
    assert base_cfg.node_range == 10_000
    assert base_cfg.min_path_length == 3 and base_cfg.max_path_length == 3

    # Increment attributes
    curriculum.increment_attr_level("degree")
    curriculum.increment_attr_level("node_range")
    curriculum.increment_attr_level("path_length")
    increased_cfg = curriculum.generate_configuration(base_value)
    assert increased_cfg.degree == 3
    assert increased_cfg.node_range == 50_000
    assert increased_cfg.min_path_length == 3 and increased_cfg.max_path_length == 5

    # Decrement degree back
    curriculum.decrement_attr_level("degree")
    partial_cfg = curriculum.generate_configuration(base_value)
    assert partial_cfg.degree == 2
    assert partial_cfg.node_range == 50_000
    assert partial_cfg.min_path_length == 3 and partial_cfg.max_path_length == 5
