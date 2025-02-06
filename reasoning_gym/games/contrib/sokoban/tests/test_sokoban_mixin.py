from unittest import TestCase

import numpy as np

from src.game import Game
from src.utils import get_state


class SokobanMixin(TestCase):
    def create_test_game(self, func):
        puzzle = func()
        np.savetxt('tmp.dat', puzzle['matrix'], fmt='%s')
        return Game(path='tmp.dat'), puzzle['state']

    def create_puzzle(self):
        matrix = np.array([['+', '+', '+', '+', '+'],
                           ['+', '-', '-', '-', '+'],
                           ['+', '*', '@', 'X', '+'],
                           ['+', '+', '+', '+', '+']])
        return {
            'matrix': matrix,
            'state': get_state(matrix)
        }

    def create_harder_puzzle(self):
        matrix = np.array([['+', '+', '+', '+', '+', '+', '+', '+'],
                           ['+', '-', '-', '@', '-', 'X', '*', '+'],
                           ['+', '+', '-', '-', '-', '+', 'X', '+'],
                           ['+', 'X', '-', '@', '-', '@', '@', '+'],
                           ['+', 'X', 'X', '@', '-', '-', '-', '+'],
                           ['+', '+', '+', '+', '+', '+', '+', '+']])
        return {
            'matrix': matrix,
            'state': get_state(matrix)
        }

    def create_invalid_puzzle(self):
        matrix = np.array([['+', '+', '+', '+', '+', '+', '+', '+'],
                           ['+', 'X', '@', '@', '-', 'X', '*', '+'],
                           ['+', '+', '-', '-', '-', '+', 'X', '+'],
                           ['+', 'X', '-', '@', '-', '@', '@', '+'],
                           ['+', 'X', 'X', '@', '-', '-', '-', '+'],
                           ['+', '+', '+', '+', '+', '+', '+', '+']])
        return {
            'matrix': matrix,
            'state': get_state(matrix)
        }

    def create_two_box_puzzle(self):
        matrix = np.array([['+', '+', '+', '+', '+'],
                           ['+', '-', '-', '-', '+'],
                           ['+', '*', '@', 'X', '+'],
                           ['+', '-', '@', 'X', '+'],
                           ['+', '+', '+', '+', '+']])
        return {
            'matrix': matrix,
            'state': get_state(matrix)
        }

    def create_walkable_puzzle(self):
        matrix = np.array([['+', '+', '+', '+', '+'],
                           ['+', '-', '-', '-', '+'],
                           ['+', '-', '-', '@', '+'],
                           ['+', '-', '*', '-', '+'],
                           ['+', '-', '-', 'X', '+'],
                           ['+', '+', '+', '+', '+']])
        return {
            'matrix': matrix,
            'state': get_state(matrix)
        }

    def create_big_puzzle(self):
        matrix = np.array([['+', '+', '+', '+', '+'],
                           ['+', '-', '-', '-', '+'],
                           ['+', '*', '@', 'X', '+'],
                           ['+', '-', '-', '@', '+'],
                           ['+', '-', '-', 'X', '+'],
                           ['+', '+', '+', '+', '+']])
        return {
            'matrix': matrix,
            'state': get_state(matrix)
        }

    def create_corner_deadlock(self):
        matrix = np.array([['+', '+', '+', '+', '+'],
                           ['+', '-', '-', '@', '+'],
                           ['+', '*', '-', 'X', '+'],
                           ['+', '-', '-', '-', '+'],
                           ['+', '+', '+', '+', '+']])
        return {
            'matrix': matrix,
            'state': get_state(matrix)
        }
    
    def create_too_many_boxes_deadlock(self):
        matrix = np.array([['+', '+', '+', '+', '+'],
                           ['+', '-', '-', '-', '+'],
                           ['+', 'X', '-', '@', '+'],
                           ['+', '*', '-', 'X', '+'],
                           ['+', '-', '-', '@', '+'],
                           ['+', '-', '-', '-', '+'],
                           ['+', '+', '+', '+', '+']])
        return {
            'matrix': matrix,
            'state': get_state(matrix)
        }

    def create_double_box_deadlock(self):
        matrix = np.array([['+', '+', '+', '+', '+'],
                           ['+', '-', '-', 'X', '+'],
                           ['+', '-', '-', '-', '+'],
                           ['+', '-', '-', '@', '+'],
                           ['+', '*', '-', '@', '+'],
                           ['+', '-', '-', '-', '+'],
                           ['+', '-', '-', 'X', '+'],
                           ['+', '+', '+', '+', '+']])
        return {
            'matrix': matrix,
            'state': get_state(matrix)
        }
