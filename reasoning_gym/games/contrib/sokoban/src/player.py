from collections import defaultdict
from random import Random

from reasoning_gym.games.contrib.sokoban.src.box import Box, Obstacle


class Player:
    """A player that can only push boxes"""

    def __init__(self, x, y, game):
        self.game = game
        self.x = x
        self.y = y

    def update(self, key: str = None) -> int:
        move = None
        if key:
            if key == "R":
                move = (1, 0)
            elif key == "L":
                move = (-1, 0)
            elif key == "U":
                move = (0, -1)
            elif key == "D":
                move = (0, 1)
        if move:
            curr = self.y, self.x
            target = self.y + move[1], self.x + move[0]
            target_elem = self.game.puzzle[target]
            if not (target_elem and target_elem.obj and isinstance(target_elem.obj, Obstacle)):
                is_box = isinstance(target_elem.obj, Box)
                if not is_box or (is_box and target_elem.obj.can_move(move)):
                    curr_elem = self.game.puzzle[curr]
                    self.y, self.x = target
                    curr_elem.char = "-" if not curr_elem.ground else "X"
                    curr_elem.obj = None
                    target_elem.char = "*" if not target_elem.ground else "%"
                    target_elem.obj = self
                    return 1
        return 0


class ReversePlayer(Player):
    """A player that can only pull boxes"""

    def __init__(self, rng: Random, x, y, game=None, puzzle=None):
        super().__init__(x=x, y=y, game=game)
        self.rng = rng
        self.game = game
        self.puzzle = puzzle
        self.curr_state = ""
        self.states = defaultdict(int)
        self.prev_move = (0, 0)

    def print_puzzle(self, matrix=None):
        print(self.puzzle_to_string(matrix=matrix))

    def puzzle_to_string(self, matrix=None):
        matrix = matrix if matrix is not None else self.game.puzzle
        height, width = len(matrix), len(matrix[0])
        ss = ""
        for h in range(height):
            for w in range(width):
                if matrix[h, w]:
                    ss = ss + str(matrix[h, w]) + " "
                else:
                    ss = ss + "F" + " "
            ss = ss + " " + "\n"
        ss = ss + "\n"
        return ss

    def get_state(self):
        state = ""
        height, width = len(self.game.puzzle), len(self.game.puzzle[0])
        for row in range(height):
            for col in range(width):
                if self.game.puzzle[row, col]:
                    state += str(self.game.puzzle[row, col])
        return state

    def update(self, puzzle_size):
        height, width = puzzle_size
        quick_chars = {
            "*": "-",
            "%": "X",
            "+": "*",
            "-": "*",
            "X": "%",
            "@": "-",
            "$": "X",
        }
        moves_tuples = [(1, 0), (-1, 0), (0, -1), (0, 1)]
        moves = self.rng.choices(moves_tuples, weights=[0.1 if m == self.prev_move else 1 for m in moves_tuples], k=1)
        self.curr_state = self.get_state()
        for move in moves:
            self.states[self.curr_state] += 1
            curr_pos = self.y, self.x
            target = self.y + move[0], self.x + move[1]
            reverse_target = self.y - move[0], self.x - move[1]
            if (
                target[1] == self.game.pad_x
                or target[0] == self.game.pad_y
                or target[1] >= self.game.pad_x + width - 1
                or target[0] >= self.game.pad_y + height - 1
                or (self.game.puzzle[target] and self.game.puzzle[target].char in "@$")
            ):
                self.prev_move = move
                return
            self.prev_move = -move[0], -move[1]
            self.game.puzzle[curr_pos].char = quick_chars[self.game.puzzle[curr_pos].char]
            self.game.puzzle[curr_pos].obj = None
            self.game.puzzle[target].char = quick_chars[self.game.puzzle[target].char]
            self.game.puzzle[target].obj = self
            if (c := self.game.puzzle[reverse_target].char) in "@$":
                self.game.puzzle[reverse_target].char = quick_chars[c]
                self.game.puzzle[reverse_target].obj.reverse_move(move)

            self.y, self.x = target
