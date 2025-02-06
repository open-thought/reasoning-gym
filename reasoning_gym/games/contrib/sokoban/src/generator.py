import random

import numpy as np
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

from reasoning_gym.games.contrib.sokoban.src.game import ReverseGame

MIN_W = 6
MIN_H = 6
MAX_W = 15
MAX_H = 10
MIN_BOXES = 4
MAX_BOXES = 10


def num_boxes(puzzle_area):
    m = (MAX_BOXES - MIN_BOXES) / (MAX_W * MAX_H - MIN_W * MIN_H)
    b = MIN_BOXES - m * MIN_W * MIN_H
    return int(m * puzzle_area + b)

def random_valid(width=10, height=10):
    return random.randrange(1, width - 1), random.randrange(1, height - 1)

def generate(window=None, seed=3, visualizer=False, path=None):
    path = path or 'levels/lvl0.dat'
    random.seed(seed)
    valid = False
    while not valid:
        width = random.randint(MIN_W, MAX_W)
        height = random.randint(MIN_H, MAX_H)
        puzzle = np.full((height, width), '+', dtype='<U1')
        boxes = num_boxes(width * height)
        boxes_seen = set()
        player_pos = random_valid(width, height)
        puzzle_size = height, width
        puzzle[player_pos[1], player_pos[0]] = '*'
        boxes_created = 0
        while boxes_created < boxes:
            box_pos = random_valid(height, width)
            if puzzle[box_pos] == '+':
                puzzle[box_pos] = '$'
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
        # player.print_puzzle(matrix)
        player.kill()
        out_of_place_boxes = np.sum([str(x) == '@' for x in matrix.flatten()])
        if out_of_place_boxes >= boxes // 2:
            # np.savetxt(path, matrix, fmt='%s')
            return reverse_game, matrix, player.puzzle_to_string(matrix)
            valid = True
            del reverse_game
        else:
            seed += 1
            del reverse_game
            # print(f'Not enough boxes out of place, generating new seed... [{out_of_place_boxes}]')


if __name__ == '__main__':
    generate()
