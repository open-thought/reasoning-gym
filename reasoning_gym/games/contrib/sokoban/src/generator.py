from random import Random

import numpy as np

from reasoning_gym.games.contrib.sokoban.src.astar import solve_astar
from reasoning_gym.games.contrib.sokoban.src.game import Game, ReverseGame


def num_boxes(puzzle_area, min_boxes, max_boxes, min_w, min_h, max_w, max_h):
    if min_w == max_w or min_h == max_h or min_boxes == max_boxes:
        return max_boxes

    m = (max_boxes - min_boxes) / (max_w * max_h - min_w * min_h)
    b = min_boxes - m * min_w * min_h
    return int(m * puzzle_area + b)


def random_valid(rng: Random, width: int = 10, height: int = 10):
    return rng.randrange(1, width - 1), rng.randrange(1, height - 1)


def generate(
    rng: Random,
    debug: bool = False,
    min_w: int = 6,
    min_h: int = 6,
    max_w: int = 15,
    max_h: int = 10,
    min_boxes: int = 4,
    max_boxes: int = 10,
    max_depth: int = 100,
    path: str = None,
) -> tuple[str, str, dict]:
    """
    Generates a level with the given configuration parameters.

    Parameters:
        rng: Random number generator
        visualizer: Whether to visualize the generation process
        min_w: Minimum width of the puzzle
        min_h: Minimum height of the puzzle
        max_w: Maximum width of the puzzle
        max_h: Maximum height of the puzzle
        min_boxes: Minimum number of boxes
        max_boxes: Maximum number of boxes
        max_depth: Maximum search depth
        path: Path to save the level file (optional)
    Returns:
        puzzle_string, solution
    """

    while True:
        width = rng.randint(min_w, max_w)
        height = rng.randint(min_h, max_h)
        puzzle = np.full((height, width), "+", dtype="<U1")
        boxes = num_boxes(width * height, min_boxes, max_boxes, min_w, min_h, max_w, max_h)
        boxes_seen = set()
        player_pos = random_valid(rng, width, height)
        puzzle_size = (height, width)
        puzzle[player_pos[1], player_pos[0]] = "*"
        boxes_created = 0
        while boxes_created < boxes:
            box_pos = random_valid(rng, height, width)
            if puzzle[box_pos] == "+":
                puzzle[box_pos] = "$"
                boxes_created += 1
                boxes_seen.add(box_pos)
        reverse_game = ReverseGame(rng=rng, width=width, height=height)
        reverse_game.load_puzzle(puzzle)
        player = reverse_game.player
        counter = round(height * width * rng.uniform(1.8, 3.6))
        while counter > 0:
            reverse_game.player.update(puzzle_size)
            if player.states[player.curr_state] >= 20:
                break
            counter -= 1
        slice_x = slice(reverse_game.pad_x, reverse_game.pad_x + width)
        slice_y = slice(reverse_game.pad_y, reverse_game.pad_y + height)
        matrix = reverse_game.puzzle[slice_y, slice_x]
        # Optionally print the puzzle:
        if debug:
            player.print_puzzle(matrix)

        out_of_place_boxes = np.sum([str(x) == "@" for x in matrix.flatten()])
        if out_of_place_boxes >= boxes // 2:
            # Optionally save the puzzle to a file:
            if path:
                np.savetxt(path, matrix, fmt="%s")
            puzzle_str = player.puzzle_to_string(matrix)

            grid_list = [list(line) for line in puzzle_str.replace(" ", "").strip().split("\n")]
            grid_array = np.array(grid_list)
            solution, depth = solve_astar(grid_array, max_depth=max_depth)
            if solution is None:
                continue  # retry generation

            if debug:
                print(f"solution={solution}")
                game = Game(width=width, height=height)
                game.load_puzzle_matrix(grid_array)

                for step, move in enumerate(solution):
                    print(f"move #{step}: {move}")
                    game.player.update(key=move)
                    game.print_puzzle()

            difficulty = {"size": puzzle_size, "num_steps": len(solution)}
            return puzzle_str, solution, difficulty
        else:
            if debug:
                print(f"Not enough boxes out of place, retrying generation... [{out_of_place_boxes}/{boxes}]")


if __name__ == "__main__":
    generate(rng=Random(), debug=True)
