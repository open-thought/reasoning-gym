"""Tests for N Queens puzzle generation"""

import pytest

from reasoning_gym.games.n_queens import NQueensConfig, NQueensDataset


def test_nqueens_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = NQueensConfig(n=-1)  # Negative not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = NQueensConfig(n=0)  # Zero not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = NQueensConfig(n=5, min_remove=5, max_remove=4)  # max < min
        config.validate()

    with pytest.raises(AssertionError):
        config = NQueensConfig(n=5, min_remove=3, max_remove=6)  # n < max
        config.validate()


def test_nqueens_dataset_deterministic():
    """Test that dataset generates same items with same seed"""
    config = NQueensConfig(seed=42, size=10)
    dataset1 = NQueensDataset(config)
    dataset2 = NQueensDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_nqueens_dataset_items():
    """Test basic properties of generated items"""
    config = NQueensConfig(n=8, min_remove=1, max_remove=7, size=10, seed=42)
    dataset = NQueensDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        # Check item structure
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata
        assert "puzzle" in item["metadata"]
        assert "solutions" in item["metadata"]
        assert "num_removed" in item["metadata"]

        puzzle = item["metadata"]["puzzle"]
        solutions = item["metadata"]["solutions"]
        num_removed = item["metadata"]["num_removed"]

        # Verify board dimensions
        assert len(puzzle) == 8
        assert all(len(row) == 8 for row in puzzle)
        for board in solutions:
            assert len(board) == 8
            assert all(len(row) == 8 for row in board)

        # Verify empty cell count
        removed_count = len(puzzle) - sum(1 for row in puzzle for cell in row if cell == "Q")
        assert config.min_remove <= removed_count <= config.max_remove
        assert removed_count == num_removed

        # Verify solution validity
        for board in solutions:
            assert is_valid_solution(board)

            # Verify puzzle matches solution where filled
            for i in range(8):
                for j in range(8):
                    if puzzle[i][j] == "Q":
                        assert puzzle[i][j] == board[i][j]


def test_nqueens_dataset_iteration():
    """Test that iteration respects dataset size"""
    config = NQueensConfig(size=5, seed=42)
    dataset = NQueensDataset(config)

    items = list(dataset)
    assert len(items) == config.size

    # Test multiple iterations yield same items
    assert items == list(dataset)


def test_nqueens_board_generation():
    """Test that generated boards are valid"""
    config = NQueensConfig(n=10, size=5, seed=42)
    dataset = NQueensDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        for board in item["metadata"]["solutions"]:
            assert is_valid_solution(board)


def test_nqueens_score_answer():
    """Test the score_answer method"""
    config = NQueensConfig(n=8, size=10, seed=42)
    dataset = NQueensDataset(config)

    # Test a few items
    for i in range(len(dataset)):
        item = dataset[i]

        # Test correct answer gets score 1.0
        valid_answer = item["metadata"]["valid_answers"][0]
        assert dataset.score_answer(valid_answer, item) == 1.0

        # Test invalid answer gets score 0.01
        invalid_answer = "_ _ _ _\n_ _ _ _\n_ _ _ _\n_ _ _ _"
        assert dataset.score_answer(invalid_answer, item) == 0.01

        # Test None answer gets score 0.0
        assert dataset.score_answer(None, item) == 0.0

    # Test python list representation of board (partial solution)
    answer = "[['_', 'Q', '_', '_'], ['_', '_', '_', 'Q'], ['Q', '_', '_', '_'], ['_', '_', 'Q', '_']]"
    entry = {"metadata": {"valid_answers": {"_ Q _ _\n_ _ _ Q\nQ _ _ _\n_ _ Q _"}}}
    assert dataset.score_answer(answer, entry) == 0.5


def is_valid_solution(board: list[list[str]]) -> bool:
    """Helper function to verify N Queens solution validity"""
    rows, cols, diags, off_diags = set(), set(), set(), set()
    n = len(board)
    num_queens = 0

    for r in range(n):
        for c in range(n):
            if board[r][c] == "Q":
                num_queens += 1
                if r in rows or c in cols or (r + c) in diags or (r - c) in off_diags:
                    return False
                rows.add(r)
                cols.add(c)
                diags.add(r + c)
                off_diags.add(r - c)

    return num_queens == n
