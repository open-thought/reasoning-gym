import pytest

from reasoning_gym.games.knight_swap import KnightSwapConfig, KnightSwapDataset, KnightSwapLogic


def test_default_config_validation():
    """Test that default configuration is valid"""
    config = KnightSwapConfig()
    config.validate()  # Should not raise any exceptions


def test_invalid_config():
    """Test that invalid configurations raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = KnightSwapConfig(min_nodes=4)  # Too few nodes
        config.validate()

    with pytest.raises(AssertionError):
        config = KnightSwapConfig(max_nodes=5, min_nodes=6)  # max < min
        config.validate()


def test_board_connectivity():
    """Test that generated boards are connected"""
    config = KnightSwapConfig(min_nodes=6, max_nodes=6, seed=42)
    dataset = KnightSwapDataset(config)
    attempts = 10
    # Try multiple puzzles since generation is random
    found_connected = False
    for i in range(attempts):
        board = dataset[i]["metadata"]["board"]
        if KnightSwapLogic.is_connected(board):
            found_connected = True
            break
        # Print debug info for failing boards
        print(f"\nBoard {i} not connected:")
        print(f"Nodes: {list(board.keys())}")
        print(f"Edges: {board}")

    assert found_connected, f"Could not find a connected board after {attempts} attempts"


def test_known_connected_board():
    """Test connectivity check with a known connected board"""
    # Create a simple connected board with valid knight moves
    board = {
        "A1": ["B3", "C2"],
        "B3": ["A1", "C1"],
        "C1": ["B3", "A2"],
        "A2": ["C1", "B4"],
        "B4": ["A2", "C2"],
        "C2": ["A1", "B4"],
    }
    assert KnightSwapLogic.is_connected(board), "Known connected board should be identified as connected"


def test_valid_knight_moves():
    """Test that all edges in generated board represent valid knight moves"""
    config = KnightSwapConfig(min_nodes=6, max_nodes=6)
    dataset = KnightSwapDataset(config)

    board = dataset[0]["metadata"]["board"]
    for start, neighbors in board.items():
        for end in neighbors:
            assert KnightSwapLogic.is_knight_move(start, end)


def test_knight_move_validation():
    """Test basic knight move validation"""
    assert KnightSwapLogic.is_knight_move("A1", "B3")  # Valid move
    assert KnightSwapLogic.is_knight_move("B3", "A1")  # Valid move reverse
    assert not KnightSwapLogic.is_knight_move("A1", "A2")  # Invalid move
    assert not KnightSwapLogic.is_knight_move("A1", "B2")  # Invalid move


def test_simple_solvable_puzzle():
    """Test a minimal solvable puzzle with one piece each"""
    config = KnightSwapConfig(min_nodes=6, max_nodes=6, min_pieces=1, max_pieces=1, impossible_ratio=0.0)
    dataset = KnightSwapDataset(config)

    # Try to find a solvable puzzle
    for i in range(5):  # Try a few times since generation is random
        puzzle = dataset[i]
        if puzzle["metadata"]["is_possible"]:
            assert puzzle["answer"] != "No"
            assert isinstance(eval(puzzle["answer"]), list)
            return

    pytest.fail("Could not find a solvable puzzle")


def test_impossible_puzzle():
    """Test that impossible puzzles are correctly identified"""
    config = KnightSwapConfig(min_nodes=6, max_nodes=6, min_pieces=2, max_pieces=2, impossible_ratio=1.0)
    dataset = KnightSwapDataset(config)

    puzzle = dataset[0]
    assert puzzle["metadata"]["is_possible"] is False
    assert puzzle["answer"] == "No"


def test_alternating_turns():
    """Test that solutions follow alternating turns rule"""
    config = KnightSwapConfig(impossible_ratio=0.0)
    dataset = KnightSwapDataset(config)

    # Find a solvable puzzle
    for i in range(5):
        puzzle = dataset[i]
        if puzzle["metadata"]["is_possible"]:
            moves = eval(puzzle["answer"])
            current_turn = puzzle["metadata"]["start_turn"]
            for move in moves:
                color = move.split(",")[0]
                assert color == current_turn
                current_turn = "B" if current_turn == "w" else "w"
            return

    pytest.fail("Could not find a solvable puzzle")


def test_solution_validation():
    """Test that solutions reach the target state"""
    config = KnightSwapConfig(impossible_ratio=0.0)
    dataset = KnightSwapDataset(config)

    # Find a solvable puzzle
    for i in range(5):
        puzzle = dataset[i]
        if puzzle["metadata"]["is_possible"]:
            # Get initial positions
            initial_white = {pos for pos, piece in puzzle["metadata"]["pieces"].items() if piece == "w"}
            initial_black = {pos for pos, piece in puzzle["metadata"]["pieces"].items() if piece == "B"}

            # Get final positions from board states
            final_state = puzzle["metadata"]["board_states"][-1]
            final_white = {pos for pos, piece in final_state.items() if piece == "w"}
            final_black = {pos for pos, piece in final_state.items() if piece == "B"}

            # Check that positions are swapped
            assert final_white == initial_black
            assert final_black == initial_white
            return

    pytest.fail("Could not find a solvable puzzle")


def test_score_calculation():
    """Test scoring for different answer types"""
    config = KnightSwapConfig()
    dataset = KnightSwapDataset(config)

    # Get a sample puzzle
    puzzle = dataset[0]

    # Test invalid answers
    assert dataset.score_answer(None, puzzle) == 0.0
    assert dataset.score_answer("", puzzle) == 0.01
    assert dataset.score_answer("Invalid", puzzle) == 0.01

    # Test correct answer
    assert dataset.score_answer(puzzle["answer"], puzzle) == 1.0
