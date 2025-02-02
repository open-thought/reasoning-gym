from random import Random
from typing import Optional, Dict, List


def gen_field(size: int, color: int = 0) -> List[int]:
    """Generate a field of given size filled with specified color (default 0)."""
    return [color] * size


def write_block(pos: int, block: List[int], field: List[int]) -> List[int]:
    """Write a block into a field at given position."""
    result = field.copy()
    for i, color in enumerate(block):
        result[pos + i] = color
    return result


def task_move_n_pix(size: int, move_pix: int, solid: bool, rng: Random) -> Optional[Dict[str, List[int]]]:
    """
    Generate a task where a block is moved to the right by move_pix pixels.

    Args:
        size: Size of the field
        move_pix: Number of pixels to move the block
        solid: If True, block is single color; if False, block has random colors
        rng: Random number generator

    Returns:
        Dictionary with 'input' and 'output' fields containing the puzzle,
        or None if valid puzzle cannot be generated
    """
    # Validate size constraints
    if size <= move_pix + 1:
        return None

    # Generate block size and position
    block_size = rng.randint(1, size - move_pix - 1)
    block_pos = rng.randint(0, size - block_size - move_pix)

    # Generate the block
    if solid:
        # For solid blocks, use single random color (1-9)
        color = rng.randint(1, 9)
        block = [color] * block_size
    else:
        # For non-solid blocks, each position gets random color (1-9)
        block = [rng.randint(1, 9) for _ in range(block_size)]

    # Create input field with block at initial position
    question = write_block(block_pos, block, gen_field(size))

    # Create output field with block moved right by move_pix
    answer = write_block(block_pos + move_pix, block, gen_field(size))

    return {"input": question, "output": answer}
