import os

from src.astar import solve_astar
from src.bfs import solve_bfs
from src.utils import is_solved
from tests.test_sokoban_mixin import SokobanMixin


class SokobanSolverTest(SokobanMixin):
    def test_bfs_correctly_solves_valid_puzzle(self):
        game, _ = self.create_test_game(self.create_harder_puzzle)
        solution, _ = solve_bfs(game.get_matrix())
        self.assertTrue(solution)
        for move in solution:
            game.player.update(key=move)
        self.assertTrue(is_solved(game.get_curr_state()))
        del game
        os.remove('tmp.dat')

    def test_a_star_correctly_solves_valid_puzzle(self):
        game, _ = self.create_test_game(self.create_harder_puzzle)
        solution, _ = solve_astar(game.get_matrix())
        self.assertTrue(solution)
        for move in solution:
            game.player.update(key=move)
        self.assertTrue(is_solved(game.get_curr_state()))
        del game
        os.remove('tmp.dat')
        
    def test_dijkstra_correctly_solves_valid_puzzle(self):
        game, _ = self.create_test_game(self.create_harder_puzzle)
        solution, _ = solve_astar(game.get_matrix())
        self.assertTrue(solution)
        for move in solution:
            game.player.update(key=move)
        self.assertTrue(is_solved(game.get_curr_state()))
        del game
        os.remove('tmp.dat')

    def test_bfs_fails_to_solve_invalid_puzzle(self):
        game, _ = self.create_test_game(self.create_invalid_puzzle)
        solution, _ = solve_bfs(game.get_matrix())
        self.assertFalse(solution)
        del game
        os.remove('tmp.dat')

    def test_a_star_fails_to_solve_invalid_puzzle(self):
        game, _ = self.create_test_game(self.create_invalid_puzzle)
        solution, _ = solve_astar(game.get_matrix())
        self.assertFalse(solution)
        del game
        os.remove('tmp.dat')

    def test_dijkstra_fails_to_solve_invalid_puzzle(self):
        game, _ = self.create_test_game(self.create_invalid_puzzle)
        solution, _ = solve_astar(game.get_matrix())
        self.assertFalse(solution)
        del game
        os.remove('tmp.dat')
