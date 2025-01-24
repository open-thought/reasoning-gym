"""Tests for mini sudoku puzzle generation"""

import pytest

from reasoning_gym.games.mini_sudoku import MiniSudokuConfig, MiniSudokuDataset


def test_mini_sudoku_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = MiniSudokuConfig(min_empty=-1)  # Negative not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = MiniSudokuConfig(min_empty=17)  # Too many empty cells
        config.validate()

    with pytest.raises(AssertionError):
        config = MiniSudokuConfig(min_empty=10, max_empty=8)  # max < min
        config.validate()


def test_mini_sudoku_dataset_deterministic():
    """Test that dataset generates same items with same seed"""
    config = MiniSudokuConfig(seed=42, size=10)
    dataset1 = MiniSudokuDataset(config)
    dataset2 = MiniSudokuDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_mini_sudoku_dataset_items():
    """Test basic properties of generated items"""
    config = MiniSudokuConfig(min_empty=8, max_empty=12, size=10, seed=42)
    dataset = MiniSudokuDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        # Check item structure
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata
        assert "puzzle" in item["metadata"]
        assert "solution" in item["metadata"]
        assert "num_empty" in item["metadata"]

        puzzle = item["metadata"]["puzzle"]
        solution = item["metadata"]["solution"]
        num_empty = item["metadata"]["num_empty"]

        # Verify board dimensions
        assert len(puzzle) == 4
        assert all(len(row) == 4 for row in puzzle)
        assert len(solution) == 4
        assert all(len(row) == 4 for row in solution)

        # Verify empty cell count
        empty_count = sum(1 for row in puzzle for cell in row if cell == 0)
        assert config.min_empty <= empty_count <= config.max_empty
        assert empty_count == num_empty

        # Verify solution validity
        assert is_valid_solution(solution)

        # Verify puzzle matches solution where filled
        for i in range(4):
            for j in range(4):
                if puzzle[i][j] != 0:
                    assert puzzle[i][j] == solution[i][j]


def test_mini_sudoku_dataset_iteration():
    """Test that iteration respects dataset size"""
    config = MiniSudokuConfig(size=5, seed=42)
    dataset = MiniSudokuDataset(config)

    items = list(dataset)
    assert len(items) == config.size

    # Test multiple iterations yield same items
    assert items == list(dataset)


def test_mini_sudoku_board_generation():
    """Test that generated boards are valid"""
    config = MiniSudokuConfig(min_empty=0, max_empty=0, size=5, seed=42)  # Force complete board
    dataset = MiniSudokuDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        board = item["metadata"]["solution"]
        assert is_valid_solution(board)


def is_valid_solution(board: list[list[int]]) -> bool:
    """Helper function to verify mini sudoku solution validity"""
    # Check rows
    for row in board:
        if set(row) != set(range(1, 5)):
            return False

    # Check columns
    for j in range(4):
        column = [board[i][j] for i in range(4)]
        if set(column) != set(range(1, 5)):
            return False

    # Check 2x2 boxes
    for box_i in range(2):
        for box_j in range(2):
            box = []
            for i in range(2):
                for j in range(2):
                    box.append(board[box_i * 2 + i][box_j * 2 + j])
            if set(box) != set(range(1, 5)):
                return False

    return True
