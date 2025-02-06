import os
import random

import numpy as np
import pygame

from reasoning_gym.games.contrib.sokoban.src.game import ReverseGame


def num_boxes(puzzle_area, min_boxes, max_boxes, min_w, min_h, max_w, max_h):
    m = (max_boxes - min_boxes) / (max_w * max_h - min_w * min_h)
    b = min_boxes - m * min_w * min_h
    return int(m * puzzle_area + b)


def random_valid(width=10, height=10):
    return random.randrange(1, width - 1), random.randrange(1, height - 1)


def generate(
    window=None, seed=3, visualizer=False, path=None, min_w=6, min_h=6, max_w=15, max_h=10, min_boxes=4, max_boxes=10
):
    """
    Generates a level with the given configuration parameters.

    Parameters:
        window: Pygame window or None.
        seed: Random seed for reproducibility.
        visualizer: Whether to visualize the generation process.
        path: Path to save the level file (default 'levels/lvl0.dat').
        min_w: Minimum width of the puzzle.
        min_h: Minimum height of the puzzle.
        max_w: Maximum width of the puzzle.
        max_h: Maximum height of the puzzle.
        min_boxes: Minimum number of boxes.
        max_boxes: Maximum number of boxes.
    Returns:
        A tuple (reverse_game, matrix, puzzle_string).
    """
    path = path or "levels/lvl0.dat"
    random.seed(seed)
    valid = False
    while not valid:
        width = random.randint(min_w, max_w)
        height = random.randint(min_h, max_h)
        puzzle = np.full((height, width), "+", dtype="<U1")
        boxes = num_boxes(width * height, min_boxes, max_boxes, min_w, min_h, max_w, max_h)
        boxes_seen = set()
        player_pos = random_valid(width, height)
        puzzle_size = (height, width)
        puzzle[player_pos[1], player_pos[0]] = "*"
        boxes_created = 0
        while boxes_created < boxes:
            box_pos = random_valid(height, width)
            if puzzle[box_pos] == "+":
                puzzle[box_pos] = "$"
                boxes_created += 1
                boxes_seen.add(box_pos)
        reverse_game = ReverseGame(window, level=0, seed=seed)
        reverse_game.load_floor()
        reverse_game.load_puzzle(puzzle)
        player = reverse_game.player
        counter = round(height * width * random.uniform(1.8, 3.6))
        while counter > 0:
            reverse_game.player.update(puzzle_size)
            if player.states[player.curr_state] >= 20:
                break
            counter -= 1
        slice_x = slice(reverse_game.pad_x, reverse_game.pad_x + width)
        slice_y = slice(reverse_game.pad_y, reverse_game.pad_y + height)
        matrix = reverse_game.puzzle[slice_y, slice_x]
        # Optionally print the puzzle:
        # player.print_puzzle(matrix)
        player.kill()
        out_of_place_boxes = np.sum([str(x) == "@" for x in matrix.flatten()])
        if out_of_place_boxes >= boxes // 2:
            # Optionally save the puzzle to a file:
            # np.savetxt(path, matrix, fmt='%s')
            valid = True
            result = (reverse_game, matrix, player.puzzle_to_string(matrix))
            return result
        else:
            seed += 1
            del reverse_game
            # print(f'Not enough boxes out of place, generating new seed... [{out_of_place_boxes}]')


if __name__ == "__main__":
    generate()
