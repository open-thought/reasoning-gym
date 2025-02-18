class Box:
    def __init__(self, x, y, game=None):
        self.game = game
        self.x = x
        self.y = y

    def can_move(self, move):
        target_x, target_y = self.x + move[0], self.y + move[1]
        target = target_y, target_x
        curr = self.y, self.x
        target_elem = self.game.puzzle[target]
        if not isinstance(target_elem.obj, Box):
            curr_elem = self.game.puzzle[curr]
            self.y, self.x = target
            curr_elem.char = "-" if not curr_elem.ground else "X"
            curr_elem.obj = None
            target_elem.char = "@" if not target_elem.ground else "$"
            target_elem.obj = self
            return True
        return False

    def reverse_move(self, move):
        target = self.y + move[0], self.x + move[1]
        curr_pos = self.y, self.x
        self.game.puzzle[curr_pos].obj = None
        self.game.puzzle[target].obj = self
        self.y, self.x = target
        self.game.puzzle[curr_pos].char = "X" if self.game.puzzle[curr_pos].ground else "-"
        self.game.puzzle[target].char = "$" if self.game.puzzle[target].ground else "@"


class Obstacle(Box):
    def __init__(self, x, y):
        super().__init__(x=x, y=y)
