import pygame
from pygame.sprite import Sprite


class Box(Sprite):
    def __init__(self, *groups, x, y, game=None):
        super().__init__(*groups)
        self.game = game
        self.sprite = pygame.image.load("reasoning_gym/games/contrib/sokoban/img/box.png")
        self.sprite = pygame.transform.scale(self.sprite, [64, 64])
        self.spriteg = pygame.image.load("reasoning_gym/games/contrib/sokoban/img/boxg.png")
        self.spriteg = pygame.transform.scale(self.spriteg, [64, 64])
        self.image = self.sprite if game and not game.puzzle[y, x].ground else self.spriteg
        self.rect = pygame.Rect(x * 64, y * 64, 64, 64)
        self.x = x
        self.y = y

    def can_move(self, move):
        target_x, target_y = self.x + move[0] // 64, self.y + move[1] // 64
        target = target_y, target_x
        curr = self.y, self.x
        target_elem = self.game.puzzle[target]
        if not isinstance(target_elem.obj, Box):
            curr_elem = self.game.puzzle[curr]
            self.rect.y, self.rect.x = target[0] * 64, target[1] * 64
            self.y, self.x = target
            curr_elem.char = "-" if not curr_elem.ground else "X"
            curr_elem.obj = None
            target_elem.char = "@" if not target_elem.ground else "$"
            target_elem.obj = self
            self.update_sprite()
            return True
        return False

    def reverse_move(self, move):
        target = self.y + move[0] // 64, self.x + move[1] // 64
        curr_pos = self.y, self.x
        self.game.puzzle[curr_pos].obj = None
        self.game.puzzle[target].obj = self
        self.rect.y, self.rect.x = target[0] * 64, target[1] * 64
        self.y, self.x = target
        self.game.puzzle[curr_pos].char = "X" if self.game.puzzle[curr_pos].ground else "-"
        self.game.puzzle[target].char = "$" if self.game.puzzle[target].ground else "@"
        self.update_sprite()

    def update_sprite(self):
        curr_obj = self.game.puzzle[self.y, self.x]
        self.image = self.spriteg if curr_obj and curr_obj.ground else self.sprite

    def __del__(self):
        self.kill()


class Obstacle(Box):
    def __init__(self, *groups, x, y):
        super().__init__(*groups, x=x, y=y)
        self.image = pygame.image.load("reasoning_gym/games/contrib/sokoban/img/obs.png")
        self.image = pygame.transform.scale(self.image, [64, 64])
        self.rect = pygame.Rect(x * 64, y * 64, 64, 64)
