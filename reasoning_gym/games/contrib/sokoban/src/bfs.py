import time
from collections import deque

import numpy as np

from reasoning_gym.games.contrib.sokoban.src.utils import can_move, get_state, is_deadlock, is_solved, print_state


def bfs(matrix, player_pos, debug=False):
    print("Breadth-First Search")
    initial_state = get_state(matrix)
    shape = matrix.shape
    print_state(initial_state, shape)
    seen = {None}
    q = deque([(initial_state, player_pos, 0, "")])
    moves = [(1, 0), (-1, 0), (0, -1), (0, 1)]
    curr_depth = 0
    direction = {
        (1, 0): "D",
        (-1, 0): "U",
        (0, -1): "L",
        (0, 1): "R",
    }
    while q:
        state, pos, depth, path = q.popleft()
        # if depth != curr_depth:
        # 	print(f'Depth: {depth}')
        # 	curr_depth = depth
        seen.add(state)
        for move in moves:
            new_state, _ = can_move(state, shape, pos, move)
            deadlock = is_deadlock(new_state, shape)
            if new_state in seen or deadlock:
                continue
            q.append(
                (
                    new_state,
                    (pos[0] + move[0], pos[1] + move[1]),
                    depth + 1,
                    path + direction[move],
                )
            )
            if is_solved(new_state):
                print(f"[BFS] Solution found!\n\n{path + direction[move]}\nDepth {depth + 1}\n")
                if debug:
                    print(f"[BFS] Solution Found!\n{path + direction[move]}", 20)
                return (path + direction[move], depth + 1)
            if debug:
                print(f"[BFS] Solution Depth: {depth + 1}\n{path + direction[move]}", 20)
    print(f"[BFS] Solution not found!\n")
    if debug:
        print(f"[BFS] Solution Not Found!\nDepth {depth + 1}", 20)
    return (None, -1 if not q else depth + 1)


def solve_bfs(puzzle, visualizer=False):
    matrix = puzzle
    where = np.where((matrix == "*") | (matrix == "%"))
    player_pos = where[0][0], where[1][0]
    return bfs(matrix, player_pos, debug=visualizer)


if __name__ == "__main__":
    start = time.time()
    root = solve_bfs(np.loadtxt("levels/lvl7.dat", dtype="<U1"))
    print(f"Runtime: {time.time() - start} seconds")
