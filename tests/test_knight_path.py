"""Tests for Knight Shortest Path task generation"""

import pytest

import reasoning_gym
from reasoning_gym.games.knight_path import KnightPathConfig, KnightPathCurriculum, KnightPathDataset


def test_knight_path_config_validation():
    """Test that invalid configurations raise assertions"""
    # Negative size
    with pytest.raises(AssertionError):
        KnightPathConfig(size=-1).validate()

    # Board size too small (< 4)
    with pytest.raises(AssertionError):
        KnightPathConfig(board_size=3).validate()

    # Board size too large (> 16)
    with pytest.raises(AssertionError):
        KnightPathConfig(board_size=17).validate()

    # Negative min_obstacles
    with pytest.raises(AssertionError):
        KnightPathConfig(min_obstacles=-1).validate()

    # min_obstacles > max_obstacles
    with pytest.raises(AssertionError):
        KnightPathConfig(min_obstacles=5, max_obstacles=2).validate()

    # max_obstacles exceeds board capacity
    with pytest.raises(AssertionError):
        KnightPathConfig(board_size=4, min_obstacles=0, max_obstacles=20).validate()


def test_knight_path_deterministic():
    """Test that dataset generates identical items with identical seeds"""
    config1 = KnightPathConfig(seed=42, size=5)
    config2 = KnightPathConfig(seed=42, size=5)

    ds1 = KnightPathDataset(config1)
    ds2 = KnightPathDataset(config2)

    for i in range(5):
        assert ds1[i] == ds2[i]


def test_knight_path_item_structure():
    """Test basic structure and validity of generated dataset items"""
    config = KnightPathConfig(board_size=8, min_obstacles=1, max_obstacles=3, seed=123, size=10)
    dataset = KnightPathDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        meta = item["metadata"]
        assert meta["source_dataset"] == "knight_path"
        assert meta["board_size"] == 8
        assert "start" in meta
        assert "target" in meta
        assert "blocked" in meta
        assert "shortest_distance" in meta

        # Verify board string is present in question
        assert "Knight start position (K):" in item["question"]
        assert "Target square (#):" in item["question"]

        # Self-score oracle answer
        assert dataset.score_answer(item["answer"], item) == 1.0


def test_knight_path_scoring():
    """Test that scoring function accurately checks valid, suboptimal, and invalid paths"""
    config = KnightPathConfig(board_size=8, min_obstacles=0, max_obstacles=0, seed=999, size=1)
    dataset = KnightPathDataset(config)
    entry = dataset[0]

    # Empty and None
    assert dataset.score_answer(None, entry) == 0.0
    assert dataset.score_answer("", entry) == 0.0
    assert dataset.score_answer("   ", entry) == 0.0

    # Exact oracle answer
    assert dataset.score_answer(entry["answer"], entry) == 1.0

    # Formatted with arrows: e.g. "c3 -> e4"
    arrow_answer = " -> ".join(entry["answer"].split())
    assert dataset.score_answer(arrow_answer, entry) == 1.0

    # Answer including starting square at front
    start_included = f"{entry['metadata']['start']} {entry['answer']}"
    assert dataset.score_answer(start_included, entry) == 1.0

    # Invalid jump (not knight movement)
    assert dataset.score_answer("a1 a2 a3", entry) == 0.0

    # Wrong destination
    assert dataset.score_answer("a1", entry) == 0.0


def test_knight_path_suboptimal_scoring():
    """Test that a valid path that takes more moves than shortest path gets 0.5 reward"""
    # Create fixed mock entry on 8x8 board from a1 to c2 (optimal 1 move: "c2")
    # Suboptimal path: a1 -> b3 -> a5 -> c4 -> a3 -> c2 (5 moves)
    entry = {
        "answer": "c2",
        "metadata": {
            "board_size": 8,
            "start": "a1",
            "target": "c2",
            "blocked": [],
            "shortest_distance": 1,
        },
    }
    config = KnightPathConfig()
    dataset = KnightPathDataset(config)

    # Suboptimal valid path
    suboptimal = "b3 a5 c4 a3 c2"
    assert dataset.score_answer(suboptimal, entry) == 0.5


def test_knight_path_blocked_square_scoring():
    """Test that stepping on a blocked square fails scoring"""
    config = KnightPathConfig()
    dataset = KnightPathDataset(config)

    # Scenario 1: Intermediate square is blocked
    entry = {
        "answer": "b3 c1 e2",
        "metadata": {
            "board_size": 8,
            "start": "a1",
            "target": "e2",
            "blocked": ["c2"],  # c2 is an obstacle
            "shortest_distance": 3,
        },
    }
    # Candidate tries to jump through blocked square c2 -> e2
    assert dataset.score_answer("c2 e2", entry) == 0.0

    # Scenario 2: Puzzle is infeasible because target is blocked
    infeasible_entry = {
        "answer": "infeasible",
        "metadata": {
            "board_size": 8,
            "start": "a1",
            "target": "c2",
            "blocked": ["c2"],
            "shortest_distance": -1,
        },
    }
    assert dataset.score_answer("c2", infeasible_entry) == 0.0
    assert dataset.score_answer("infeasible", infeasible_entry) == 1.0


def test_knight_path_curriculum():
    """Test curriculum registration and configuration"""
    curriculum = KnightPathCurriculum()
    assert curriculum is not None
    assert "max_obstacles" in curriculum.attributes
    assert "board_size" in curriculum.attributes


def test_knight_path_factory_registration():
    """Test creating dataset through reasoning_gym.create_dataset factory"""
    ds = reasoning_gym.create_dataset("knight_path", size=3, seed=777)
    assert len(ds) == 3
    assert ds[0]["metadata"]["source_dataset"] == "knight_path"
