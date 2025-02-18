import pytest

from reasoning_gym.games.rush_hour import Board, RushHourConfig, RushHourDataset


def test_rush_hour_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = RushHourConfig(min_moves=0)
        config.validate()

    with pytest.raises(AssertionError):
        config = RushHourConfig(min_moves=3, max_moves=2)
        config.validate()

    with pytest.raises(AssertionError):
        config = RushHourConfig(size=0)
        config.validate()


def test_rush_hour_deterministic():
    """Test that dataset generates same items with same seed"""
    config = RushHourConfig(seed=42, size=10, min_moves=1, max_moves=50)
    dataset1 = RushHourDataset(config)
    dataset2 = RushHourDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i]["metadata"] == dataset2[i]["metadata"]


def test_rush_hour_items():
    """Test basic properties of generated items"""
    config = RushHourConfig(min_moves=1, max_moves=10, size=10, seed=42)
    dataset = RushHourDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Verify metadata contains required fields
        assert "board_config" in item["metadata"]
        assert "min_moves" in item["metadata"]

        # Verify min_moves is within configured range
        assert config.min_moves <= item["metadata"]["min_moves"] <= config.max_moves

        # Verify board_config is valid length
        assert len(item["metadata"]["board_config"]) == 36  # 6x6 board


def test_rush_hour_move_filtering():
    """Test that puzzles are filtered by move count"""
    config = RushHourConfig(min_moves=5, max_moves=10, size=10, seed=42)
    dataset = RushHourDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        moves = item["metadata"]["min_moves"]
        assert 5 <= moves <= 10, f"Puzzle with {moves} moves outside configured range 5-10"


def test_score_answer():
    """Test that score_answer correctly validates solutions"""
    config = RushHourConfig(min_moves=1, max_moves=50, size=10, seed=42)
    dataset = RushHourDataset(config)

    # Get a puzzle
    puzzle = dataset[0]

    # Test invalid answers
    assert dataset.score_answer(None, puzzle) == 0.0
    assert dataset.score_answer("", puzzle) == 0.0
    assert dataset.score_answer("invalid", puzzle) == 0.0
    assert dataset.score_answer("A+1 B-2 INVALID", puzzle) == 0.0

    # Test incomplete solution
    assert dataset.score_answer("A+1 B-2", puzzle) == 0.0


def test_perform_moves():
    b = Board("GBBoLoGHIoLMGHIAAMCCCKoMooJKDDEEJFFo")
    assert not b.solved
    incomplete_moves = "F+1 K+1 M-1 C+3 H+2 J-1 E+1 G+3 B-1 I-1 A-3 I+1 L+1 B+3 I-1 A+2 G-3"
    b.perform_moves(incomplete_moves)
    assert not b.solved
    solution = "E-1 H-3 A-1 J+1 C-3 M+1 B+1 K-4 A+1 C+2 D-1 F-1 H+3 A-1 K+1 B-1 M-1 C+1 J-1 E+1 G+3 A-1 I+1 B-3 I-1 A+1 G-1 E-1 J+1 C-1 K-1 L-1 M+3 A+3"
    b.perform_moves(solution)
    assert b.solved


def test_perform_moves_walls():
    b = Board("BBoIKxCCCIKoGAAJooGoHJDDooHEELoFFoxL")
    print(b.board_str())
    # assert sum(1 for p in b._pieces if p.fixed) == 2, "two walls expected"
    # assert not b.solved

    # b.perform_moves(
    #     "F-1 G+1 A-1 H-1 E-2 J+2 D-1 L-3 D+1 J-2 E+3 H+2 A+1 J+2 D-3 I+2 K+2 B+3 L+1 C+3 G-3 A-1 D-1 H-4 A+1 D+1 F+1 G+4 A-1 D-1 H+2 B-2"
    # )
    # assert not b.solved

    # b.perform_moves(
    #     "C-3 I-2 J-2 E-3 J+2 I+2 B+2 C+3 H-2 A+1 D+1 G-4 A-1 D-1 E-1 F-1 H+4 A+1 B-2 D+1 G+2 C-3 I-2 K-2 L+1 A+3"
    # )
    # assert b.solved
