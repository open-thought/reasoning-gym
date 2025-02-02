from random import Random
from typing import Dict, List, Optional


def gen_field(size: int, color: int = 0) -> List[int]:
    """Generate a field of given size filled with specified color (default 0)."""
    return [color] * size


def write_block(pos: int, block: List[int], field: List[int]) -> List[int]:
    """Write a block into a field at given position."""
    result = field.copy()
    for i, color in enumerate(block):
        result[pos + i] = color
    return result


def task_move_n_pix(rng: Random, size: int, move_pix: int, solid: bool) -> Optional[Dict[str, List[int]]]:
    """Generate a task where a block is moved to the right by move_pix pixels."""
    if size <= move_pix + 1:
        return None

    block_size = rng.randint(1, size - move_pix - 1)
    block_pos = rng.randint(0, size - block_size - move_pix)

    if solid:
        color = rng.randint(1, 9)
        block = [color] * block_size
    else:
        block = [rng.randint(1, 9) for _ in range(block_size)]

    question = write_block(block_pos, block, gen_field(size))
    answer = write_block(block_pos + move_pix, block, gen_field(size))

    return {"input": question, "output": answer}


def task_move_n_pix_wrapped(rng: Random, size: int, move_pix: int, solid: bool) -> Optional[Dict[str, List[int]]]:
    """Generate a task where a block is moved to the right by move_pix pixels with wrapping."""
    block_size = rng.randint(1, size)
    block_pos = rng.randint(0, size)

    if solid:
        color = rng.randint(1, 9)
        block = [color] * block_size
    else:
        block = [rng.randint(1, 9) for _ in range(block_size)]

    question = gen_field(size)
    answer = gen_field(size)

    for i, color in enumerate(block):
        question[(block_pos + i) % size] = color
        answer[(block_pos + move_pix + i) % size] = color

    return {"input": question, "output": answer}


def task_gravity(rng: Random, size: int) -> Optional[Dict[str, List[int]]]:
    """Generate a task where all non-zero elements are attracted to the left."""
    density = 0.5
    question = [rng.randint(1, 9) if rng.random() < density else 0 for _ in range(size)]

    non_zero = [x for x in question if x != 0]
    answer = non_zero + [0] * (size - len(non_zero))

    return {"input": question, "output": answer}


def task_gravity_counting(rng: Random, size: int) -> Optional[Dict[str, List[int]]]:
    """Generate a task where non-zero elements are counted and represented as a sequence of 1s."""
    density = 0.5
    question = [rng.randint(1, 9) if rng.random() < density else 0 for _ in range(size)]

    count = sum(1 for x in question if x != 0)
    answer = [1] * count + [0] * (size - count)

    return {"input": question, "output": answer}


def task_gravity_antigravity(rng: Random, size: int) -> Optional[Dict[str, List[int]]]:
    """Generate a task where color 1 moves right and color 2 moves left."""
    density = 0.5
    question = [rng.randint(1, 2) if rng.random() < density else 0 for _ in range(size)]

    color1 = [x for x in question if x == 1]
    color2 = [x for x in question if x == 2]
    answer = [2] * len(color2) + [0] * (size - len(color1) - len(color2)) + [1] * len(color1)

    return {"input": question, "output": answer}


def task_block_touch_dot(rng: Random, size: int) -> Optional[Dict[str, List[int]]]:
    """Generate a task where a block moves to touch (but not cover) a dot."""
    dot_color = 1
    block_color = rng.randint(2, 9)

    block_size = rng.randint(1, size)
    dot_pos = rng.randint(0, size)

    can_place_left = dot_pos >= block_size
    can_place_right = dot_pos + block_size < size

    if not (can_place_left or can_place_right):
        return None

    if can_place_left and can_place_right:
        side = rng.choice(["left", "right"])
    elif can_place_left:
        side = "left"
    else:
        side = "right"

    if side == "left":
        q_block_pos = rng.randint(0, dot_pos - block_size)
        a_block_pos = dot_pos - block_size
    else:
        q_block_pos = rng.randint(dot_pos + 1, size - block_size)
        a_block_pos = dot_pos + 1

    question = gen_field(size)
    question[dot_pos] = dot_color
    question = write_block(q_block_pos, [block_color] * block_size, question)

    answer = gen_field(size)
    answer[dot_pos] = dot_color
    answer = write_block(a_block_pos, [block_color] * block_size, answer)

    return {"input": question, "output": answer}


def task_block_touch_dot_n_pix(rng: Random, size: int, move_pix: int) -> Optional[Dict[str, List[int]]]:
    """Generate a task where a block moves move_pix pixels toward a dot."""
    dot_color = 2
    block_color = rng.randint(3, 9)

    block_size = rng.randint(1, size)
    dot_pos = rng.randint(0, size)

    can_place_left = dot_pos >= block_size
    can_place_right = dot_pos + block_size < size

    if not (can_place_left or can_place_right):
        return None

    if can_place_left and can_place_right:
        side = rng.choice(["left", "right"])
    elif can_place_left:
        side = "left"
    else:
        side = "right"

    if side == "left":
        q_block_pos = rng.randint(0, dot_pos - block_size)
        distance = (dot_pos - block_size) - q_block_pos
        move = min(distance, move_pix)
        a_block_pos = q_block_pos + move
    else:
        q_block_pos = rng.randint(dot_pos + 1, size - block_size)
        distance = q_block_pos - (dot_pos + 1)
        move = min(distance, move_pix)
        a_block_pos = q_block_pos - move

    question = gen_field(size)
    question[dot_pos] = dot_color
    question = write_block(q_block_pos, [block_color] * block_size, question)

    answer = gen_field(size)
    answer[dot_pos] = dot_color
    answer = write_block(a_block_pos, [block_color] * block_size, answer)

    return {"input": question, "output": answer}


def task_block_scale_to_dot(rng: Random, size: int) -> Optional[Dict[str, List[int]]]:
    """Generate a task where a block scales to touch a dot (keeping one end fixed)."""
    dot_color = 2
    block_color = rng.randint(3, 9)

    block_size = rng.randint(1, size)
    dot_pos = rng.randint(0, size)

    can_place_left = dot_pos >= block_size
    can_place_right = dot_pos + block_size < size

    if not (can_place_left or can_place_right):
        return None

    if can_place_left and can_place_right:
        side = rng.choice(["left", "right"])
    elif can_place_left:
        side = "left"
    else:
        side = "right"

    if side == "left":
        q_block_pos = rng.randint(0, dot_pos - block_size)
        new_size = dot_pos - q_block_pos + 1
        a_block_pos = q_block_pos
    else:
        q_block_pos = rng.randint(dot_pos + 1, size - block_size)
        new_size = (q_block_pos + block_size) - dot_pos
        a_block_pos = dot_pos

    question = gen_field(size)
    question[dot_pos] = dot_color
    question = write_block(q_block_pos, [block_color] * block_size, question)

    answer = gen_field(size)
    answer[dot_pos] = dot_color
    answer = write_block(a_block_pos, [block_color] * new_size, answer)

    return {"input": question, "output": answer}


def task_two_points_and_fill(rng: Random, size: int) -> Optional[Dict[str, List[int]]]:
    """Generate a task where space between two points of same color is filled with that color."""
    color = rng.randint(1, 9)

    pos1 = rng.randint(0, size - 1)
    pos2 = rng.randint(0, size - 1)
    if pos1 == pos2:
        return None

    pos1, pos2 = min(pos1, pos2), max(pos1, pos2)

    question = gen_field(size)
    question[pos1] = color
    question[pos2] = color

    answer = question.copy()
    for i in range(pos1, pos2 + 1):
        answer[i] = color

    return {"input": question, "output": answer}


def task_reflect_block_with_border_pixel(rng: Random, size: int) -> Optional[Dict[str, List[int]]]:
    """Generate a task where a block with a border pixel is reflected."""
    block_size = rng.randint(2, size)
    if block_size > size:
        return None

    c1 = rng.randint(1, 9)
    c2 = rng.randint(1, 9)
    if c1 == c2:
        return None

    side = "left" if rng.random() < 0.5 else "right"
    pos = rng.randint(0, size - block_size)

    block = [c1] * block_size
    if side == "left":
        block[0] = c2
    else:
        block[block_size - 1] = c2

    question = write_block(pos, block, gen_field(size))
    reversed_block = block[::-1]  # Reverse the block
    answer = write_block(pos, reversed_block, gen_field(size))

    return {"input": question, "output": answer}


def task_reflect_block_with_border_pixel_random(rng: Random, size: int) -> Optional[Dict[str, List[int]]]:
    """Generate a task where a random-colored block with a border pixel is reflected."""
    block_size = rng.randint(2, size)
    if block_size > size:
        return None

    side = "left" if rng.random() < 0.5 else "right"
    pos = rng.randint(0, size - block_size)

    block = [rng.randint(1, 9) for _ in range(block_size)]
    border_color = rng.randint(1, 9)

    if side == "left":
        if block[0] == border_color:
            return None
        block[0] = border_color
    else:
        if block[block_size - 1] == border_color:
            return None
        block[block_size - 1] = border_color

    question = write_block(pos, block, gen_field(size))
    reversed_block = block[::-1]  # Reverse the block
    answer = write_block(pos, reversed_block, gen_field(size))

    return {"input": question, "output": answer}


def task_reflect_block_around_dot(rng: Random, size: int) -> Optional[Dict[str, List[int]]]:
    """Generate a task where a block is reflected around a dot."""
    dot_color = 2

    dot_pos = rng.randint(0, size)
    block_size = rng.randint(1, size)
    block_pos = rng.randint(0, size - block_size)
    block_end = block_pos + block_size - 1

    # Check if block is strictly to left or right of dot
    strictly_left = block_end < dot_pos
    strictly_right = block_pos > dot_pos

    if not (strictly_left or strictly_right):
        return None

    block_color = rng.randint(3, 9)  # Different from dot color
    block = [block_color] * block_size

    # Calculate reflection bounds
    min_reflect = 2 * dot_pos - block_end
    max_reflect = 2 * dot_pos - block_pos
    if min_reflect < 0 or max_reflect >= size:
        return None

    question = gen_field(size)
    question = write_block(block_pos, block, question)
    question[dot_pos] = dot_color

    answer = gen_field(size)
    answer[dot_pos] = dot_color
    for i in range(block_size):
        reflect_idx = 2 * dot_pos - (block_pos + i)
        answer[reflect_idx] = block[i]

    return {"input": question, "output": answer}


def task_block_and_noise_remove(rng: Random, size: int) -> Optional[Dict[str, List[int]]]:
    """Generate a task where noise around a block needs to be removed."""
    block_size = rng.randint(2, size)
    if block_size > size:
        return None

    block_pos = rng.randint(0, size - block_size)
    color = rng.randint(1, 9)

    # Create field with block
    field = gen_field(size)
    for i in range(block_size):
        field[block_pos + i] = color

    # Track forbidden positions for noise
    forbidden = [False] * size
    for i in range(block_pos, block_pos + block_size):
        forbidden[i] = True
    if block_pos > 0:
        forbidden[block_pos - 1] = True
    if block_pos + block_size < size:
        forbidden[block_pos + block_size] = True

    # Add noise
    noise_count = rng.randint(1, 3)
    noise_positions = []

    for _ in range(noise_count):
        allowed = [i for i in range(size) if not forbidden[i]]
        if not allowed:
            break
        noise_pos = rng.choice(allowed)
        noise_positions.append(noise_pos)
        field[noise_pos] = color
        forbidden[noise_pos] = True
        if noise_pos > 0:
            forbidden[noise_pos - 1] = True
        if noise_pos + 1 < size:
            forbidden[noise_pos + 1] = True

    if len(noise_positions) < noise_count:
        return None

    question = field
    answer = field.copy()
    for pos in noise_positions:
        answer[pos] = 0

    return {"input": question, "output": answer}


def task_block_and_noise_remove_inside(rng: Random, size: int) -> Optional[Dict[str, List[int]]]:
    """Generate a task where noise inside a block needs to be removed."""
    if size <= 6:
        return None

    block_size = rng.randint(6, size)
    if block_size > size:
        return None

    block_pos = rng.randint(0, size - block_size)
    color = rng.randint(1, 9)

    # Create field with block
    field = gen_field(size)
    for i in range(block_size):
        field[block_pos + i] = color

    # Add noise inside block
    max_noise = max(1, (block_size // 2) - 1)
    noise_count = rng.randint(1, max_noise)

    positions = list(range(block_size))
    rng.shuffle(positions)
    noise_positions = positions[:noise_count]

    for offset in noise_positions:
        pos = block_pos + offset
        noise_color = rng.randint(1, 9)
        while noise_color == color:
            noise_color = rng.randint(1, 9)
        field[pos] = noise_color

    question = field
    answer = field.copy()
    for offset in noise_positions:
        answer[block_pos + offset] = color

    return {"input": question, "output": answer}


def task_copy_block_to_dots(rng: Random, size: int) -> Optional[Dict[str, List[int]]]:
    """Generate a task where a block pattern is copied to dot positions."""
    block_size = 3 if rng.random() < 0.5 else 5
    if block_size >= size:
        return None

    color = rng.randint(1, 9)
    block = [color] * block_size

    # Generate dots with minimum distance to prevent overlap
    min_gap = block_size
    dot_positions = []
    pos = block_size + block_size // 2 + 1

    while pos <= size - block_size:
        if rng.random() < 0.5:  # Control dot density
            dot_positions.append(pos)
            pos += min_gap
        pos += 1

    if not dot_positions:
        return None

    question = gen_field(size)
    question = write_block(0, block, question)
    for pos in dot_positions:
        question[pos] = color

    answer = gen_field(size)
    answer = write_block(0, block, answer)
    for pos in dot_positions:
        block_start = pos - block_size // 2
        answer = write_block(block_start, block, answer)

    return {"input": question, "output": answer}


def task_copy_block_to_dots_colors(rng: Random, size: int) -> Optional[Dict[str, List[int]]]:
    """Generate a task where a block pattern is copied to dot positions with matching colors."""
    block_size = 3 if rng.random() < 0.5 else 5
    if block_size >= size:
        return None

    block_color = rng.randint(1, 9)
    block = [block_color] * block_size

    # Generate dots with minimum distance to prevent overlap
    min_gap = block_size
    dot_positions = []
    dot_colors = []
    pos = block_size + block_size // 2 + 1

    while pos < size - block_size:
        if rng.random() < 0.5:
            dot_color = rng.randint(1, 9)
            dot_positions.append(pos)
            dot_colors.append(dot_color)
            pos += min_gap
        pos += 1

    if not dot_positions:
        return None

    question = gen_field(size)
    question = write_block(0, block, question)
    for i, pos in enumerate(dot_positions):
        question[pos] = dot_colors[i]

    answer = gen_field(size)
    answer = write_block(0, block, answer)
    for i, pos in enumerate(dot_positions):
        block_start = pos - block_size // 2
        colored_block = [dot_colors[i]] * block_size
        answer = write_block(block_start, colored_block, answer)

    return {"input": question, "output": answer}


def task_paint_biggest_block(rng: Random, size: int) -> Optional[Dict[str, List[int]]]:
    """Generate a task where the largest block is painted a different color."""
    target_color = 1
    initial_color = rng.randint(2, 9)

    # Generate random blocks
    question = gen_field(size)
    blocks = []
    pos = 0

    while pos < size:
        if rng.random() < 0.4 and size - pos >= 2:
            block_size = rng.randint(2, min(size - pos, 6))
            blocks.append((pos, block_size))
            for i in range(block_size):
                question[pos + i] = initial_color
            pos += block_size + 1
        else:
            pos += 1

    if len(blocks) < 2:
        return None

    # Find biggest block
    biggest_pos, biggest_size = max(blocks, key=lambda x: x[1])

    # Check if there are multiple blocks of the same size
    biggest_count = sum(1 for _, size in blocks if size == biggest_size)
    if biggest_count > 1:
        return None

    answer = question.copy()
    for i in range(biggest_size):
        answer[biggest_pos + i] = target_color

    return {"input": question, "output": answer}


def task_sort_blocks_by_size(rng: Random, size: int) -> Optional[Dict[str, List[int]]]:
    """Generate a task where blocks are sorted by size with 1 pixel gaps."""
    color = rng.randint(1, 9)
    blocks = []
    pos = 0

    # Generate random blocks with random sizes
    while pos < size:
        if rng.random() < 0.4 and size - pos >= 2:
            block_size = rng.randint(1, min(size - pos, 6))
            blocks.append((pos, block_size))
            pos += block_size + rng.randint(1, 4)  # Random gaps
        else:
            pos += 1

    if len(blocks) < 2:
        return None

    # Create input field
    question = gen_field(size)
    for pos, block_size in blocks:
        for i in range(block_size):
            question[pos + i] = color

    # Sort blocks by size
    blocks.sort(key=lambda x: x[1])

    # Check if sorted blocks fit with gaps
    total_space = sum(size for _, size in blocks) + len(blocks) - 1
    if total_space > size:
        return None

    # Create answer field with sorted blocks
    answer = gen_field(size)
    current_pos = 0

    for _, block_size in blocks:
        for i in range(block_size):
            answer[current_pos + i] = color
        current_pos += block_size + 1  # One pixel gap

    return {"input": question, "output": answer}


def task_sort_complete_sequence(rng: Random, size: int) -> Optional[Dict[str, List[int]]]:
    """Generate a task where a complete sequence of block sizes is sorted."""
    # Calculate max possible block size given total array size
    max_size = 1
    total_space = 0
    while total_space + max_size + 1 <= size:
        total_space += max_size + 1
        max_size += 1
    max_size -= 1

    if max_size < 2:
        return None

    color = rng.randint(1, 9)

    # Create sequence of all sizes from 1 to max_size
    blocks = list(range(1, max_size + 1))
    rng.shuffle(blocks)

    # Create input field with shuffled blocks
    question = gen_field(size)
    pos = 0
    for block_size in blocks:
        for i in range(block_size):
            question[pos + i] = color
        pos += block_size + 1

    # Create answer field with sorted blocks
    answer = gen_field(size)
    pos = 0
    for block_size in range(1, max_size + 1):
        for i in range(block_size):
            answer[pos + i] = color
        pos += block_size + 1

    return {"input": question, "output": answer}


def task_recolor_blocks_by_size(rng: Random, size: int) -> Optional[Dict[str, List[int]]]:
    """Generate a task where two blocks are recolored based on their size."""
    # Generate two different random sizes
    size1 = rng.randint(2, 8)
    size2 = rng.randint(2, 8)
    while size2 == size1:
        size2 = rng.randint(2, 8)

    # Ensure both blocks fit with at least 1 gap
    if size1 + size2 + 1 > size:
        return None

    # Place blocks with gap
    pos1 = rng.randint(0, size - (size1 + size2 + 1))
    pos2 = rng.randint(pos1 + size1 + 1, size - size2)

    # Create input field with both blocks color 3
    question = gen_field(size)
    for i in range(size1):
        question[pos1 + i] = 3
    for i in range(size2):
        question[pos2 + i] = 3

    # Create answer field with recolored blocks
    answer = question.copy()
    if size1 > size2:
        for i in range(size1):
            answer[pos1 + i] = 1
        for i in range(size2):
            answer[pos2 + i] = 2
    else:
        for i in range(size1):
            answer[pos1 + i] = 2
        for i in range(size2):
            answer[pos2 + i] = 1

    return {"input": question, "output": answer}


def task_gravity_one_step(rng: Random, size: int) -> Optional[Dict[str, List[int]]]:
    """Generate a task where non-zero elements move one step left if possible."""
    question = [rng.randint(1, 9) if rng.random() < 0.5 else 0 for _ in range(size)]
    answer = question.copy()

    # Move each non-zero pixel one step left if possible
    for i in range(1, size):
        if answer[i] != 0 and answer[i - 1] == 0:
            answer[i - 1] = answer[i]
            answer[i] = 0

    return {"input": question, "output": answer}


def task_move_block_by_own_size(rng: Random, size: int) -> Optional[Dict[str, List[int]]]:
    """Generate a task where a block moves right by its own size."""
    block_size = rng.randint(1, size // 2)  # Ensure space for movement
    pos = rng.randint(0, size - block_size * 2)  # Space for block and movement
    color = rng.randint(1, 9)

    question = gen_field(size)
    block = [color] * block_size
    question = write_block(pos, block, question)

    answer = write_block(pos + block_size, block, gen_field(size))

    return {"input": question, "output": answer}


def task_change_to_five(rng: Random, size: int) -> Optional[Dict[str, List[int]]]:
    """Generate a task where all non-zero colors change to 5."""
    density = 0.5
    question = [rng.randint(1, 9) if rng.random() < density else 0 for _ in range(size)]
    answer = [5 if x != 0 else 0 for x in question]

    return {"input": question, "output": answer}


def task_recolor_blocks_from_palette(rng: Random, size: int) -> Optional[Dict[str, List[int]]]:
    """Generate a task where blocks are recolored using a color palette."""
    # Generate blocks of same size
    block_size = rng.randint(2, 4)
    blocks = []
    pos = 0

    while pos + block_size <= size:
        if rng.random() < 0.4:
            blocks.append(pos)
            pos += block_size + 1
        else:
            pos += 1

    # Ensure we have space for palette
    while blocks and blocks[-1] + block_size + len(blocks) + 1 >= size:
        blocks.pop()

    if not blocks:
        return None

    # Shift blocks right to make room for palette
    palette_size = len(blocks)
    blocks = [pos + palette_size + 1 for pos in blocks]

    # Generate color palette
    colors = []
    for _ in range(len(blocks)):
        while True:
            color = rng.randint(1, 9)
            if color not in colors:
                colors.append(color)
                break

    # Create question with color palette and blocks
    question = gen_field(size)

    # Place color palette at start
    for i, color in enumerate(colors):
        question[i] = color

    # Place blocks of color 5
    for block_pos in blocks:
        for i in range(block_size):
            question[block_pos + i] = 5

    # Create answer with recolored blocks
    answer = question.copy()
    for block_idx, block_pos in enumerate(blocks):
        color = colors[block_idx]
        for i in range(block_size):
            answer[block_pos + i] = color

    return {"input": question, "output": answer}


def task_duplicate_block_from_seeds(rng: Random, size: int) -> Optional[Dict[str, List[int]]]:
    """Generate a task where a block is duplicated from seed pixels."""
    block_size = rng.randint(2, 4)
    if block_size + 1 >= size:
        return None
    if size <= 3 + block_size:
        return None

    # Position block with space for seeds
    block_pos = rng.randint(2, size - block_size - 1)

    # Decide seed placement
    left_seed = rng.random() < 0.5
    right_seed = rng.random() < 0.5
    if not (left_seed or right_seed):
        return None

    # Create input
    question = gen_field(size)

    # Place main block
    for i in range(block_size):
        question[block_pos + i] = 1

    # Place seeds with gaps
    seeds = []
    if left_seed:
        color = rng.randint(1, 9)
        question[block_pos - 2] = color
        seeds.append(("left", block_pos - 2, color))
    if right_seed:
        color = rng.randint(1, 9)
        question[block_pos + block_size + 1] = color
        seeds.append(("right", block_pos + block_size + 1, color))

    # Create answer with duplicated blocks
    answer = question.copy()

    for side, seed_pos, color in seeds:
        if side == "left":
            # For left seed, blocks end at seed
            end_pos = seed_pos
            while end_pos >= 0:
                start_pos = end_pos - block_size + 1
                for pos in range(max(0, start_pos), end_pos + 1):
                    answer[pos] = color
                if start_pos < 1:
                    break
                end_pos = start_pos - 2  # -1 for gap
        else:  # side == "right"
            # For right seed, blocks start at seed
            start_pos = seed_pos
            while start_pos < size:
                for offset in range(min(block_size, size - start_pos)):
                    answer[start_pos + offset] = color
                if start_pos + block_size + 1 >= size:
                    break
                start_pos = start_pos + block_size + 1  # +1 for gap

    return {"input": question, "output": answer}


def task_fill_from_pixel(rng: Random, size: int) -> Optional[Dict[str, List[int]]]:
    """Generate a task where a pixel fills in one direction until hitting another pixel."""
    block_size = rng.randint(3, 6)
    if block_size >= size - 2:
        return None

    # Position block with space for seed
    block_pos = rng.randint(1, size - block_size - 1)

    # Create input
    question = gen_field(size)

    # Place main block
    block_color = rng.randint(1, 9)
    for i in range(block_size):
        question[block_pos + i] = block_color

    # Place seed pixel and determine fill direction
    seed_color = rng.randint(1, 9)
    while seed_color == block_color:
        seed_color = rng.randint(1, 9)

    is_left = rng.random() < 0.5

    if is_left:
        question[block_pos - 1] = seed_color
    else:
        question[block_pos + block_size] = seed_color

    # Create answer with fill
    answer = question.copy()

    if is_left:
        # Fill from seed to left border
        for i in range(block_pos):
            answer[i] = seed_color
    else:
        # Fill from seed to right border
        for i in range(block_pos + block_size, size):
            answer[i] = seed_color

    return {"input": question, "output": answer}


def task_mark_size_two_blocks(rng: Random, size: int) -> Optional[Dict[str, List[int]]]:
    """Generate a task where size-2 blocks are marked with surrounding pixels."""
    blocks = []
    pos = 0

    # Generate blocks with minimum gap of 2
    while pos < size:
        if rng.random() < 0.4:
            block_size = rng.randint(1, 3)
            # Check if we have space for block and potential markers
            needed_space = block_size + (2 if block_size == 2 else 0)
            if pos + needed_space < size:
                blocks.append((pos, block_size))
                pos += block_size + 2  # Minimum gap of 2

        pos += 1

    if len(blocks) < 2:
        return None

    # Verify gaps between blocks (including markers)
    valid = True
    for i in range(len(blocks) - 1):
        pos1, size1 = blocks[i]
        pos2, _ = blocks[i + 1]
        needed_gap = 3 if size1 == 2 else 2
        if pos2 - (pos1 + size1) < needed_gap:
            valid = False
            break
    if not valid:
        return None

    # Create input with blocks
    question = gen_field(size)
    for pos, block_size in blocks:
        # Place block
        for i in range(block_size):
            question[pos + i] = 1

    # Create answer with markers
    answer = question.copy()
    for pos, block_size in blocks:
        if block_size == 2:
            # Add markers for size 2 blocks
            if pos > 0:
                answer[pos - 1] = 3
            if pos + block_size < size:
                answer[pos + block_size] = 3

    return {"input": question, "output": answer}


def task_fill_until_collision(rng: Random, size: int) -> Optional[Dict[str, List[int]]]:
    """Generate a task where pixels fill empty space until collision."""
    # At least 4 positions for meaningful puzzle
    if size < 4:
        return None

    is_left = rng.random() < 0.5
    question = gen_field(size)

    # Place the side marker
    if is_left:
        question[0] = 5
    else:
        question[size - 1] = 5

    # Place 2-4 random pixels
    num_pixels = rng.randint(2, 4)
    positions = []

    if is_left:
        # Skip first position
        for _ in range(num_pixels):
            while True:
                pos = rng.randint(1, size - 1)
                if pos not in positions:
                    positions.append(pos)
                    break
    else:
        # Skip last position
        for _ in range(num_pixels):
            while True:
                pos = rng.randint(0, size - 2)
                if pos not in positions:
                    positions.append(pos)
                    break

    # Color random pixels
    for pos in positions:
        question[pos] = rng.randint(1, 9)

    positions.sort()

    # Create answer
    answer = question.copy()

    if is_left:
        # Fill right from each pixel
        prev_pos = 0  # Start from marker
        for pos in positions:
            color = question[pos]
            # Fill from previous position to current
            for i in range(prev_pos + 1, pos):
                answer[i] = color
            prev_pos = pos
    else:
        # Fill left from each pixel
        prev_pos = size - 1  # Start from marker
        for pos in reversed(positions):
            color = question[pos]
            # Fill from current position to previous
            for i in range(pos + 1, prev_pos):
                answer[i] = color
            prev_pos = pos

    return {"input": question, "output": answer}


def task_repeat_pattern_full(rng: Random, size: int) -> Optional[Dict[str, List[int]]]:
    """Generate a task where a pattern is repeated to fill the space."""
    # Generate initial pattern
    pattern_size = rng.randint(2, 5)
    pattern = [rng.randint(1, 9) for _ in range(pattern_size)]

    # Calculate total size needed for 2 repetitions
    double_size = pattern_size * 2
    if double_size >= size:
        return None

    # Create input with 2 repetitions
    question = gen_field(size)
    for i in range(pattern_size):
        question[i] = pattern[i]
        question[i + pattern_size] = pattern[i]

    # Create answer with maximum repetitions
    answer = gen_field(size)
    pos = 0
    while pos + pattern_size <= size:
        for i in range(pattern_size):
            answer[pos + i] = pattern[i]
        pos += pattern_size

    # Fill remaining space (if any) with pattern elements
    for i in range(pos, size):
        answer[i] = pattern[i - pos]

    return {"input": question, "output": answer}


def task_gravity_weighted_colors(rng: Random, size: int) -> Optional[Dict[str, List[int]]]:
    """Generate a task where color 2 is heavier than color 1 in gravity."""
    # Generate random field with only colors 1 and 2
    question = [rng.randint(1, 2) if rng.random() < 0.5 else 0 for _ in range(size)]

    # Count colors
    count_1 = sum(1 for x in question if x == 1)
    count_2 = sum(1 for x in question if x == 2)

    # Create answer with sorted colors
    answer = gen_field(size)

    # Place heavier color 2 first
    for i in range(count_2):
        answer[i] = 2

    # Then place color 1
    for i in range(count_1):
        answer[count_2 + i] = 1

    return {"input": question, "output": answer}


def task_color_left_half_blocks(rng: Random, size: int) -> Optional[Dict[str, List[int]]]:
    """Generate a task where left half of blocks are colored differently."""
    pos = 0
    question = gen_field(size)
    blocks = []

    # Generate blocks with gap 1
    while pos < size:
        if rng.random() < 0.4:
            block_size = rng.randint(2, 8)
            if pos + block_size >= size:
                break

            blocks.append((pos, block_size))
            for i in range(block_size):
                question[pos + i] = 2
            pos += block_size + 1  # block size + gap
        else:
            pos += 1

    if len(blocks) < 2:
        return None

    # Create answer with half-colored blocks
    answer = question.copy()
    for pos, block_size in blocks:
        half_size = block_size // 2
        for i in range(half_size):
            answer[pos + i] = 8

    return {"input": question, "output": answer}


def task_mirror(task_result: Optional[Dict[str, List[int]]]) -> Optional[Dict[str, List[int]]]:
    """Mirror the input and output arrays of a task result."""
    if task_result is None:
        return None
    return {"input": list(reversed(task_result["input"])), "output": list(reversed(task_result["output"]))}


def task_inverse(task_result: Optional[Dict[str, List[int]]]) -> Optional[Dict[str, List[int]]]:
    """Swap the input and output arrays of a task result."""
    if task_result is None:
        return None
    return {"input": task_result["output"], "output": task_result["input"]}


def task_identity(task_result: Optional[Dict[str, List[int]]]) -> Optional[Dict[str, List[int]]]:
    """Return the task result unchanged."""
    return task_result


# Table of all ARC 1D task functions with their parameters
ARC_1D_TASKS = {
    # Move tasks - right direction
    "move_1pix_solid_right": (task_move_n_pix, {"move_pix": 1, "solid": True}),
    "move_2pix_solid_right": (task_move_n_pix, {"move_pix": 2, "solid": True}),
    "move_3pix_solid_right": (task_move_n_pix, {"move_pix": 3, "solid": True}),
    "move_4pix_solid_right": (task_move_n_pix, {"move_pix": 4, "solid": True}),
    "move_1pix_colorful_right": (task_move_n_pix, {"move_pix": 1, "solid": False}),
    "move_2pix_colorful_right": (task_move_n_pix, {"move_pix": 2, "solid": False}),
    "move_3pix_colorful_right": (task_move_n_pix, {"move_pix": 3, "solid": False}),
    "move_4pix_colorful_right": (task_move_n_pix, {"move_pix": 4, "solid": False}),
    # Move tasks - left direction (mirrored)
    "move_1pix_solid_left": (
        lambda rng, size, **kwargs: task_mirror(task_move_n_pix(rng, size, **kwargs)),
        {"move_pix": 1, "solid": True},
    ),
    "move_2pix_solid_left": (
        lambda rng, size, **kwargs: task_mirror(task_move_n_pix(rng, size, **kwargs)),
        {"move_pix": 2, "solid": True},
    ),
    "move_3pix_solid_left": (
        lambda rng, size, **kwargs: task_mirror(task_move_n_pix(rng, size, **kwargs)),
        {"move_pix": 3, "solid": True},
    ),
    "move_4pix_solid_left": (
        lambda rng, size, **kwargs: task_mirror(task_move_n_pix(rng, size, **kwargs)),
        {"move_pix": 4, "solid": True},
    ),
    "move_1pix_colorful_left": (
        lambda rng, size, **kwargs: task_mirror(task_move_n_pix(rng, size, **kwargs)),
        {"move_pix": 1, "solid": False},
    ),
    "move_2pix_colorful_left": (
        lambda rng, size, **kwargs: task_mirror(task_move_n_pix(rng, size, **kwargs)),
        {"move_pix": 2, "solid": False},
    ),
    "move_3pix_colorful_left": (
        lambda rng, size, **kwargs: task_mirror(task_move_n_pix(rng, size, **kwargs)),
        {"move_pix": 3, "solid": False},
    ),
    "move_4pix_colorful_left": (
        lambda rng, size, **kwargs: task_mirror(task_move_n_pix(rng, size, **kwargs)),
        {"move_pix": 4, "solid": False},
    ),
    # Move wrapped tasks - right direction
    "move_1pix_solid_wrapped_right": (task_move_n_pix_wrapped, {"move_pix": 1, "solid": True}),
    "move_2pix_solid_wrapped_right": (task_move_n_pix_wrapped, {"move_pix": 2, "solid": True}),
    "move_3pix_solid_wrapped_right": (task_move_n_pix_wrapped, {"move_pix": 3, "solid": True}),
    "move_4pix_solid_wrapped_right": (task_move_n_pix_wrapped, {"move_pix": 4, "solid": True}),
    "move_1pix_colorful_wrapped_right": (task_move_n_pix_wrapped, {"move_pix": 1, "solid": False}),
    "move_2pix_colorful_wrapped_right": (task_move_n_pix_wrapped, {"move_pix": 2, "solid": False}),
    "move_3pix_colorful_wrapped_right": (task_move_n_pix_wrapped, {"move_pix": 3, "solid": False}),
    "move_4pix_colorful_wrapped_right": (task_move_n_pix_wrapped, {"move_pix": 4, "solid": False}),
    # Move wrapped tasks - left direction (mirrored)
    "move_1pix_solid_wrapped_left": (
        lambda rng, size, **kwargs: task_mirror(task_move_n_pix_wrapped(rng, size, **kwargs)),
        {"move_pix": 1, "solid": True},
    ),
    "move_2pix_solid_wrapped_left": (
        lambda rng, size, **kwargs: task_mirror(task_move_n_pix_wrapped(rng, size, **kwargs)),
        {"move_pix": 2, "solid": True},
    ),
    "move_3pix_solid_wrapped_left": (
        lambda rng, size, **kwargs: task_mirror(task_move_n_pix_wrapped(rng, size, **kwargs)),
        {"move_pix": 3, "solid": True},
    ),
    "move_4pix_solid_wrapped_left": (
        lambda rng, size, **kwargs: task_mirror(task_move_n_pix_wrapped(rng, size, **kwargs)),
        {"move_pix": 4, "solid": True},
    ),
    "move_1pix_colorful_wrapped_left": (
        lambda rng, size, **kwargs: task_mirror(task_move_n_pix_wrapped(rng, size, **kwargs)),
        {"move_pix": 1, "solid": False},
    ),
    "move_2pix_colorful_wrapped_left": (
        lambda rng, size, **kwargs: task_mirror(task_move_n_pix_wrapped(rng, size, **kwargs)),
        {"move_pix": 2, "solid": False},
    ),
    "move_3pix_colorful_wrapped_left": (
        lambda rng, size, **kwargs: task_mirror(task_move_n_pix_wrapped(rng, size, **kwargs)),
        {"move_pix": 3, "solid": False},
    ),
    "move_4pix_colorful_wrapped_left": (
        lambda rng, size, **kwargs: task_mirror(task_move_n_pix_wrapped(rng, size, **kwargs)),
        {"move_pix": 4, "solid": False},
    ),
    # Gravity tasks - right direction
    "gravity_right": (task_gravity, {}),
    "gravity_counting_right": (task_gravity_counting, {}),
    "gravity_antigravity_right": (task_gravity_antigravity, {}),
    "gravity_one_step_right": (task_gravity_one_step, {}),
    "gravity_weighted_colors_right": (task_gravity_weighted_colors, {}),
    # Gravity tasks - left direction (mirrored)
    "gravity_left": (lambda rng, size, **kwargs: task_mirror(task_gravity(rng, size, **kwargs)), {}),
    "gravity_counting_left": (lambda rng, size, **kwargs: task_mirror(task_gravity_counting(rng, size, **kwargs)), {}),
    "gravity_antigravity_left": (
        lambda rng, size, **kwargs: task_mirror(task_gravity_antigravity(rng, size, **kwargs)),
        {},
    ),
    "gravity_one_step_left": (lambda rng, size, **kwargs: task_mirror(task_gravity_one_step(rng, size, **kwargs)), {}),
    "gravity_weighted_colors_left": (
        lambda rng, size, **kwargs: task_mirror(task_gravity_weighted_colors(rng, size, **kwargs)),
        {},
    ),
    # Block tasks
    "block_touch_dot": (task_block_touch_dot, {}),
    "block_touch_dot_1pix": (task_block_touch_dot_n_pix, {"move_pix": 1}),
    "block_touch_dot_2pix": (task_block_touch_dot_n_pix, {"move_pix": 2}),
    "block_touch_dot_3pix": (task_block_touch_dot_n_pix, {"move_pix": 3}),
    "block_touch_dot_4pix": (task_block_touch_dot_n_pix, {"move_pix": 4}),
    "block_scale_to_dot": (task_block_scale_to_dot, {}),
    "block_and_noise_remove": (task_block_and_noise_remove, {}),
    "block_and_noise_remove_inside": (task_block_and_noise_remove_inside, {}),
    "move_block_by_own_size": (task_move_block_by_own_size, {}),
    # Pattern tasks
    "two_points_and_fill": (task_two_points_and_fill, {}),
    "two_points_and_fill_inv": (
        lambda rng, size, **kwargs: task_inverse(task_two_points_and_fill(rng, size, **kwargs)),
        {},
    ),
    "copy_block_to_dots": (task_copy_block_to_dots, {}),
    "copy_block_to_dots_colors": (task_copy_block_to_dots_colors, {}),
    "repeat_pattern_full": (task_repeat_pattern_full, {}),
    # Reflection tasks
    "reflect_block_with_border_pixel": (task_reflect_block_with_border_pixel, {}),
    "reflect_block_random": (task_reflect_block_with_border_pixel_random, {}),
    "reflect_block_around_dot": (task_reflect_block_around_dot, {}),
    # Color tasks
    "paint_biggest_block": (task_paint_biggest_block, {}),
    "recolor_blocks_by_size": (task_recolor_blocks_by_size, {}),
    "change_to_five": (task_change_to_five, {}),
    "recolor_blocks_from_palette": (task_recolor_blocks_from_palette, {}),
    "color_left_half_blocks": (task_color_left_half_blocks, {}),
    # Sorting tasks
    "sort_blocks_by_size": (task_sort_blocks_by_size, {}),
    "sort_complete_sequence": (task_sort_complete_sequence, {}),
    # Fill tasks
    "duplicate_block_from_seeds": (task_duplicate_block_from_seeds, {}),
    "fill_from_pixel": (task_fill_from_pixel, {}),
    "fill_until_collision": (task_fill_until_collision, {}),
    # Marking tasks
    "mark_size_two_blocks": (task_mark_size_two_blocks, {}),
}
