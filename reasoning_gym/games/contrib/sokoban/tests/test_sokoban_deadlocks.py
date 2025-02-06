import os
from unittest import skip

from src.utils import is_deadlock
from tests.test_sokoban_mixin import SokobanMixin


class SokobanDeadlockTest(SokobanMixin):
    def test_box_on_corner_deadlock_is_detected(self):
        game, state = self.create_test_game(self.create_corner_deadlock)
        self.assertTrue(is_deadlock(state, (game.puzzle_size)))
        del game
        os.remove('tmp.dat')

    def test_too_many_boxes_on_border_deadlock_is_detected(self):
        game, state = self.create_test_game(self.create_too_many_boxes_deadlock)
        self.assertTrue(is_deadlock(state, (game.puzzle_size)))
        del game
        os.remove('tmp.dat')

    def test_double_box_deadlock_is_detected(self):
        game, state = self.create_test_game(self.create_double_box_deadlock)
        self.assertTrue(is_deadlock(state, (game.puzzle_size)))
        del game
        os.remove('tmp.dat')
