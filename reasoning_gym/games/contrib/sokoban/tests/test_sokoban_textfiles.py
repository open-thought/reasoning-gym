import os

import numpy as np

from src.game import Game
from tests.test_sokoban_mixin import SokobanMixin


class SokobanTextFilesTest(SokobanMixin):
    def setUp(self) -> None:
        return super().setUp()

    def test_the_test(self):
        assert 1

    def test_puzzle_is_correctly_loaded_from_textfile(self):
        game, _ = self.create_test_game(self.create_puzzle)
        self.assertEqual(game.puzzle_size, (4, 5))
        self.assertEqual(len(game.object_group), 16)
        player_pos = game.pad_y + 2, game.pad_x + 1
        self.assertEqual(player_pos, (game.player.y, game.player.x))
        del game
        os.remove('tmp.dat')
