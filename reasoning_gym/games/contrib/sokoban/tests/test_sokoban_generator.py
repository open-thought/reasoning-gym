import os
import random

import pytest

from src.game import Game
from src.generator import generate
from src.utils import find_boxes_and_goals, is_solved
from tests.test_sokoban_mixin import SokobanMixin


@pytest.mark.slow
class SokobanGeneratorTest(SokobanMixin):
    def test_randomly_generated_puzzle_is_valid(self):
        random.seed(1)
        i = 0
        while i < 30:
            generate(random.randint(1, 99999), path='tmp.dat')
            game = Game(path='tmp.dat')
            boxes, goals, _ = find_boxes_and_goals(game.get_curr_state(), game.puzzle_size)
            self.assertEqual(len(boxes), len(goals))
            del game
            i += 1
        os.remove('tmp.dat')

    def test_randomly_generated_board_is_not_solved(self):
        random.seed(1)
        i = 0
        while i < 30:
            generate(random.randint(1, 99999), path='tmp.dat')
            game = Game(path='tmp.dat')
            self.assertFalse(is_solved(game.get_curr_state()))
            del game
            i += 1
        os.remove('tmp.dat')