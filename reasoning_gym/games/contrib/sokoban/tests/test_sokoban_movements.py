import os

from src.utils import get_state, is_solved
from tests.test_sokoban_mixin import SokobanMixin


class SokobanMovementTest(SokobanMixin):
    def test_player_correctly_moves_to_the_right(self):
        game, _ = self.create_test_game(self.create_walkable_puzzle)
        target_state = '++++++---++--@++--*++--X++++++'
        game.player.update(key='R')
        self.assertEqual(game.get_curr_state(), target_state)
        os.remove('tmp.dat')

    def test_player_correctly_moves_to_the_left(self):
        game, _ = self.create_test_game(self.create_walkable_puzzle)
        target_state = '++++++---++--@++*--++--X++++++'
        game.player.update(key='L')
        self.assertEqual(game.get_curr_state(), target_state)
        os.remove('tmp.dat')

    def test_player_correctly_moves_up(self):
        game, _ = self.create_test_game(self.create_walkable_puzzle)
        target_state = '++++++---++-*@++---++--X++++++'
        game.player.update(key='U')
        self.assertEqual(game.get_curr_state(), target_state)
        os.remove('tmp.dat')

    def test_player_correctly_moves_down(self):
        game, _ = self.create_test_game(self.create_walkable_puzzle)
        target_state = '++++++---++--@++---++-*X++++++'
        game.player.update(key='D')
        self.assertEqual(game.get_curr_state(), target_state)
        os.remove('tmp.dat')

    def test_player_can_move_boxes(self):
        game, _ = self.create_test_game(self.create_two_box_puzzle)
        target_state = '++++++---++-*$++-@X++++++'
        game.player.update(key='R')
        self.assertEqual(game.get_curr_state(), target_state)
        os.remove('tmp.dat')

    def test_player_cannot_go_out_of_bounds(self):
        game, _ = self.create_test_game(self.create_puzzle)
        target_state = '++++++--*++-@X++++++'
        game.player.update(key='U')
        game.player.update(key='U')
        game.player.update(key='U')
        game.player.update(key='R')
        game.player.update(key='R')
        game.player.update(key='R')
        self.assertEqual(game.get_curr_state(), target_state)
        os.remove('tmp.dat')
        
    def test_player_can_solve_the_puzzle(self):
        game, _ = self.create_test_game(self.create_puzzle)
        target_state = '++++++---++-*$++++++'
        game.player.update(key='R')
        curr_state = game.get_curr_state()
        self.assertEqual(curr_state, target_state)
        self.assertTrue(is_solved(curr_state))
        os.remove('tmp.dat')


