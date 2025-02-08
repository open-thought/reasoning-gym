"""Tests for Tower of Hanoi puzzle generation"""

import re

import pytest

from reasoning_gym.games.tower_of_hanoi import HanoiConfig, HanoiDataset


def test_toh_config_validation():
    """Test that invalid configurations raise appropriate errors."""
    # Test negative number of disks
    with pytest.raises(AssertionError):
        config = HanoiConfig(min_disks=0)  # At least 1 disk required
        config.validate()

    # Test max_disks less than min_disks
    with pytest.raises(AssertionError):
        config = HanoiConfig(min_disks=5, max_disks=3)
        config.validate()

    # Test min_pegs less than 3
    with pytest.raises(AssertionError):
        config = HanoiConfig(min_pegs=2)
        config.validate()

    # Test max_pegs less than min_pegs
    with pytest.raises(AssertionError):
        config = HanoiConfig(min_pegs=3, max_pegs=2)
        config.validate()

    # Test invalid move configurations if any (assuming such validations exist)
    # Add more tests based on the actual validation logic in HanoiConfig


def test_toh_dataset_deterministic():
    """Test that dataset generates the same items with the same seed."""
    config = HanoiConfig(seed=42, size=10)
    dataset1 = HanoiDataset(config)
    dataset2 = HanoiDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i], f"Mismatch found in instance {i} with seed 42."


def test_toh_dataset_items():
    """Test basic properties of generated items."""
    config = HanoiConfig(min_disks=3, max_disks=5, min_pegs=3, max_pegs=4, size=10, seed=42)
    dataset = HanoiDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]

        # Check item structure
        assert isinstance(item, dict), f"Item {i} is not a dictionary."
        assert "question" in item, f"Item {i} missing 'question' key."
        assert "answer" in item, f"Item {i} missing 'answer' key."
        assert "metadata" in item, f"Item {i} missing 'metadata' key."

        # Check metadata
        metadata = item["metadata"]
        assert "num_disks" in metadata, f"Item {i} metadata missing 'num_disks'."
        assert "num_pegs" in metadata, f"Item {i} metadata missing 'num_pegs'."
        assert "start_peg" in metadata, f"Item {i} metadata missing 'start_peg'."
        assert "target_peg" in metadata, f"Item {i} metadata missing 'target_peg'."
        assert "auxiliary_pegs" in metadata, f"Item {i} metadata missing 'auxiliary_pegs'."
        assert "solution_length" in metadata, f"Item {i} metadata missing 'solution_length'."

        num_disks = metadata["num_disks"]
        num_pegs = metadata["num_pegs"]
        start_peg = metadata["start_peg"]
        target_peg = metadata["target_peg"]
        auxiliary_pegs = metadata["auxiliary_pegs"]
        solution_length = metadata["solution_length"]

        # Verify peg counts
        assert num_pegs == len(metadata["auxiliary_pegs"]) + 2, f"Item {i} has inconsistent peg counts."

        # Verify solution_length consistency
        assert solution_length == len(
            item["answer"]
        ), f"Item {i} metadata 'solution_length' does not match actual number of moves."

        # Optional: Additional checks like verifying that start and target pegs are distinct
        assert start_peg != target_peg, f"Item {i} has identical start and target pegs."


def test_toh_move_validity():
    """Test that all moves in each problem instance are valid according to Tower of Hanoi rules."""
    config = HanoiConfig(min_disks=3, max_disks=5, min_pegs=3, max_pegs=4, size=10, seed=42)
    dataset = HanoiDataset(config)

    for idx, instance in enumerate(dataset):
        num_disks = instance["metadata"]["num_disks"]
        num_pegs = instance["metadata"]["num_pegs"]
        start_peg = instance["metadata"]["start_peg"]
        target_peg = instance["metadata"]["target_peg"]
        auxiliary_pegs = instance["metadata"]["auxiliary_pegs"]
        pegs = list(range(1, num_pegs + 1))

        # Initialize pegs_state: all disks start on the start peg
        pegs_state = {peg: [] for peg in pegs}
        for disk in range(num_disks, 0, -1):
            pegs_state[start_peg].append(disk)

        # Iterate over each move and validate
        for move_num, move in enumerate(instance["answer"], start=1):
            disk, from_peg, to_peg = parse_move(move)

            # Check that from_peg exists
            assert from_peg in pegs, f"Move {move_num} in Instance {idx} references non-existent from_peg {from_peg}."

            # Check that to_peg exists
            assert to_peg in pegs, f"Move {move_num} in Instance {idx} references non-existent to_peg {to_peg}."

            # Check that from_peg is not empty
            assert pegs_state[
                from_peg
            ], f"Move {move_num} in Instance {idx} attempts to move from an empty Peg {from_peg}."

            # Check that the disk to move is on top of from_peg
            top_disk = pegs_state[from_peg][-1]
            assert disk == top_disk, (
                f"Move {move_num} in Instance {idx} attempts to move disk {disk} "
                f"which is not on top of Peg {from_peg} (top disk: {top_disk})."
            )

            # Check that moving disk to to_peg does not violate size constraints
            if pegs_state[to_peg]:
                top_to_disk = pegs_state[to_peg][-1]
                assert top_to_disk > disk, (
                    f"Move {move_num} in Instance {idx} attempts to place disk {disk} "
                    f"on top of smaller disk {top_to_disk} on Peg {to_peg}."
                )

            # Perform the move
            pegs_state[from_peg].pop()
            pegs_state[to_peg].append(disk)


def test_toh_final_state_correct():
    """Test that the final state of each problem instance has all disks on the target peg in correct order."""
    config = HanoiConfig(min_disks=3, max_disks=5, min_pegs=3, max_pegs=4, size=10, seed=42)
    dataset = HanoiDataset(config)

    for idx, instance in enumerate(dataset):
        num_disks = instance["metadata"]["num_disks"]
        num_pegs = instance["metadata"]["num_pegs"]
        start_peg = instance["metadata"]["start_peg"]
        target_peg = instance["metadata"]["target_peg"]
        auxiliary_pegs = instance["metadata"]["auxiliary_pegs"]
        pegs = list(range(1, num_pegs + 1))

        # Initialize pegs_state: all disks start on the start peg
        pegs_state = {peg: [] for peg in pegs}
        for disk in range(num_disks, 0, -1):
            pegs_state[start_peg].append(disk)

        # Perform all moves
        for move in instance["answer"]:
            disk, from_peg, to_peg = parse_move(move)
            pegs_state[from_peg].pop()
            pegs_state[to_peg].append(disk)

        # After all moves, all disks should be on target peg in descending order
        final_pegs = pegs_state[target_peg]
        assert len(final_pegs) == num_disks, f"Instance {idx} does not have all disks on the target Peg {target_peg}."

        # Verify that disks are in correct order on target peg
        expected_final = list(range(num_disks, 0, -1))
        assert final_pegs == expected_final, f"Instance {idx} has disks on Peg {target_peg} in incorrect order."

        # Ensure all other pegs are empty
        for peg in pegs:
            if peg != target_peg:
                assert (
                    len(pegs_state[peg]) == 0
                ), f"Instance {idx} has disks remaining on Peg {peg}, which should be empty."


def test_toh_dataset_iteration():
    """Test that iteration respects dataset size and multiple iterations yield the same items."""
    config = HanoiConfig(min_disks=3, max_disks=5, min_pegs=3, max_pegs=4, size=5, seed=42)
    dataset = HanoiDataset(config)

    # Test dataset size
    assert len(dataset) == config.size, f"Dataset size mismatch: expected {config.size}, got {len(dataset)}."

    # Collect items
    items = list(dataset)

    # Test multiple iterations yield the same items
    assert items == list(dataset), "Multiple iterations over the dataset do not yield the same items."


def parse_move(move_str: str) -> tuple:
    """Parse a move string and extract disk number, from peg, and to peg.

    Args:
        move_str (str): Move instruction, e.g., "Move disk 2 from Peg 1 to Peg 3".

    Returns:
        tuple: (disk, from_peg, to_peg)
    """
    pattern = r"Move disk (\d+) from Peg (\d+) to Peg (\d+)"
    match = re.match(pattern, move_str)
    assert match is not None, f"Move string '{move_str}' does not match the expected format."
    disk = int(match.group(1))
    from_peg = int(match.group(2))
    to_peg = int(match.group(3))
    return disk, from_peg, to_peg


def is_valid_final_state(pegs_state: dict, target_peg: int, num_disks: int) -> bool:
    """Verify that all disks are on the target peg in descending order.

    Args:
        pegs_state (dict): Current state of the pegs.
        target_peg (int): The target peg number.
        num_disks (int): Total number of disks.

    Returns:
        bool: True if valid, False otherwise.
    """
    target_stack = pegs_state[target_peg]
    if len(target_stack) != num_disks:
        return False
    return target_stack == list(range(num_disks, 0, -1))


def test_score_answer():
    """
    Test that the score_answer method returns the expected reward values.

    Expected behavior:
      - Correct answer (i.e. equivalent in length, or better, than the one provided in the dataset item) gives 1.0.
      - A correct solution that is suboptimal length gives a proportional reward of optimal_move_count/user_move_count
      - A badly formatted answer gives a minimal reward (0.01).
      - An answer that is syntactically valid but does not solve the puzzle gives a partial reward (0.05).
      - An empty string gives 0.01.
      - None gives 0.0.
    """
    # Create a dataset instance using the default configuration.
    config = HanoiConfig(min_disks=3, max_disks=5, min_pegs=3, max_pegs=4, size=5, seed=42)
    dataset = HanoiDataset(config)
    # Pick one instance from the dataset for testing.
    item = dataset[0]
    correct_answer = item["answer"]

    # 1. Correct answer should yield full reward.
    score_correct = dataset.score_answer(answer=correct_answer, entry=item)
    assert score_correct == 1.0, f"Correct answer score {score_correct} is not 1.0."

    # 2. A badly formatted answer should yield minimal reward (0.01).
    score_bad_format = dataset.score_answer(answer="a wrong solution", entry=item)
    assert score_bad_format == 0.01, f"Badly formatted answer score {score_bad_format} is not 0.01."

    # 3. An answer that is validly formatted but unsolved.
    # For example, remove the last move from the correct answer.
    unfinished_answer = correct_answer[:-1]
    score_unsolved = dataset.score_answer(answer=unfinished_answer, entry=item)
    assert score_unsolved == 0.05, f"Unsolved answer score {score_unsolved} is not 0.05."

    # 4. An empty answer should yield 0.01.
    score_empty = dataset.score_answer(answer="", entry=item)
    assert score_empty == 0.01, f"Empty answer score {score_empty} is not 0.01."

    # 5. A None answer should yield 0.0.
    score_none = dataset.score_answer(answer=None, entry=item)
    assert score_none == 0.0, f"None answer score {score_none} is not 0.0."
