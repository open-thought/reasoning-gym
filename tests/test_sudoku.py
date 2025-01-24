"""Tests for sudoku puzzle generation"""

import pytest

from reasoning_gym.games.sudoku import SudokuConfig, SudokuDataset


def test_sudoku_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = SudokuConfig(min_empty=-1)  # Negative not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = SudokuConfig(min_empty=82)  # Too many empty cells
        config.validate()

    with pytest.raises(AssertionError):
        config = SudokuConfig(min_empty=50, max_empty=40)  # max < min
        config.validate()


def test_sudoku_dataset_deterministic():
    """Test that dataset generates same items with same seed"""
    config = SudokuConfig(seed=42, size=10)
    dataset1 = SudokuDataset(config)
    dataset2 = SudokuDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_sudoku_dataset_items():
    """Test basic properties of generated items"""
    config = SudokuConfig(min_empty=30, max_empty=40, size=10, seed=42)
    dataset = SudokuDataset(config)

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
        assert len(puzzle) == 9
        assert all(len(row) == 9 for row in puzzle)
        assert len(solution) == 9
        assert all(len(row) == 9 for row in solution)

        # Verify empty cell count
        empty_count = sum(1 for row in puzzle for cell in row if cell == 0)
        assert config.min_empty <= empty_count <= config.max_empty
        assert empty_count == num_empty

        # Verify solution validity
        assert is_valid_solution(solution)

        # Verify puzzle matches solution where filled
        for i in range(9):
            for j in range(9):
                if puzzle[i][j] != 0:
                    assert puzzle[i][j] == solution[i][j]


def test_sudoku_dataset_iteration():
    """Test that iteration respects dataset size"""
    config = SudokuConfig(size=5, seed=42)
    dataset = SudokuDataset(config)

    items = list(dataset)
    assert len(items) == config.size

    # Test multiple iterations yield same items
    assert items == list(dataset)


def test_sudoku_board_generation():
    """Test that generated boards are valid"""
    config = SudokuConfig(min_empty=0, max_empty=0, size=5, seed=42)  # Force complete board
    dataset = SudokuDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        board = item["metadata"]["solution"]
        assert is_valid_solution(board)


def is_valid_solution(board: list[list[int]]) -> bool:
    """Helper function to verify sudoku solution validity"""
    # Check rows
    for row in board:
        if set(row) != set(range(1, 10)):
            return False

    # Check columns
    for j in range(9):
        column = [board[i][j] for i in range(9)]
        if set(column) != set(range(1, 10)):
            return False

    # Check 3x3 boxes
    for box_i in range(3):
        for box_j in range(3):
            box = []
            for i in range(3):
                for j in range(3):
                    box.append(board[box_i * 3 + i][box_j * 3 + j])
            if set(box) != set(range(1, 10)):
                return False

    return True
