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

def task_move_n_pix_wrapped(size: int, move_pix: int, solid: bool, rng: Random) -> Optional[Dict[str, List[int]]]:
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

def task_gravity(size: int, rng: Random) -> Optional[Dict[str, List[int]]]:
    """Generate a task where all non-zero elements are attracted to the left."""
    density = 0.5
    question = [rng.randint(1, 9) if rng.random() < density else 0 for _ in range(size)]
    
    non_zero = [x for x in question if x != 0]
    answer = non_zero + [0] * (size - len(non_zero))
    
    return {"input": question, "output": answer}

def task_gravity_counting(size: int, rng: Random) -> Optional[Dict[str, List[int]]]:
    """Generate a task where non-zero elements are counted and represented as a sequence of 1s."""
    density = 0.5
    question = [rng.randint(1, 9) if rng.random() < density else 0 for _ in range(size)]
    
    count = sum(1 for x in question if x != 0)
    answer = [1] * count + [0] * (size - count)
    
    return {"input": question, "output": answer}

def task_gravity_antigravity(size: int, rng: Random) -> Optional[Dict[str, List[int]]]:
    """Generate a task where color 1 moves right and color 2 moves left."""
    density = 0.5
    question = [rng.randint(1, 2) if rng.random() < density else 0 for _ in range(size)]
    
    color1 = [x for x in question if x == 1]
    color2 = [x for x in question if x == 2]
    answer = [2] * len(color2) + [0] * (size - len(color1) - len(color2)) + [1] * len(color1)
    
    return {"input": question, "output": answer}

def task_block_touch_dot(size: int, rng: Random) -> Optional[Dict[str, List[int]]]:
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

def task_block_touch_dot_n_pix(size: int, move_pix: int, rng: Random) -> Optional[Dict[str, List[int]]]:
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

def task_block_scale_to_dot(size: int, rng: Random) -> Optional[Dict[str, List[int]]]:
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

def task_two_points_and_fill(size: int, rng: Random) -> Optional[Dict[str, List[int]]]:
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

def task_reflect_block_with_border_pixel(size: int, rng: Random) -> Optional[Dict[str, List[int]]]:
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

def task_reflect_block_with_border_pixel_random(size: int, rng: Random) -> Optional[Dict[str, List[int]]]:
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

def task_reflect_block_around_dot(size: int, rng: Random) -> Optional[Dict[str, List[int]]]:
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

def task_block_and_noise_remove(size: int, rng: Random) -> Optional[Dict[str, List[int]]]:
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

def task_block_and_noise_remove_inside(size: int, rng: Random) -> Optional[Dict[str, List[int]]]:
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

def task_copy_block_to_dots(size: int, rng: Random) -> Optional[Dict[str, List[int]]]:
    """Generate a task where a block pattern is copied to dot positions."""
    block_size = 3 if rng.random() < 0.5 else 5
    if block_size >= size:
        return None
        
    color = rng.randint(1, 9)
    block = [color] * block_size
    
    # Generate dots with minimum distance to prevent overlap
    min_gap = block_size
    dot_positions = []
    pos = block_size + block_size//2 + 1
    
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
        block_start = pos - block_size//2
        answer = write_block(block_start, block, answer)
        
    return {"input": question, "output": answer}
