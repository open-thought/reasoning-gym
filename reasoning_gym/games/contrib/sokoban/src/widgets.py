# import pygame
# from pygame_widgets.button import Button
# from pygame_widgets.textbox import TextBox
# from pygame_widgets.toggle import Toggle

# from reasoning_gym.games.contrib.sokoban.src.events import *


# def sidebar_widgets(window):
# 	prev_button = Button(
# 		window, 1030, 12, 22, 40, text='<', radius=2,
# 		font=pygame.font.SysFont('Verdana', 18, bold=True),
# 		onClick=lambda: pygame.event.post(pygame.event.Event(PREVIOUS_EVENT)),
# 		borderColor='black', borderThickness=2,
# 	)
# 	label = Label(window, f'Level 0', 1055, 10, 30)
# 	next_button = Button(
# 		window, 1188, 12, 22, 40, text='>', radius=2,
# 		font=pygame.font.SysFont('Verdana', 18, bold=True),
# 		onClick=lambda: pygame.event.post(pygame.event.Event(NEXT_EVENT)),
# 		borderColor='black', borderThickness=2,
# 	)
# 	restart = Button(
# 		window, 1055, 130, 130, 40, text='Restart', radius=5,
# 		font=pygame.font.SysFont('Verdana', 18, bold=True),
# 		onClick=lambda: pygame.event.post(pygame.event.Event(RESTART_EVENT)),
# 		borderColor='black', borderThickness=2,
# 	)
# 	random_game = Button(
# 		window, 1055, 220, 130, 40, text='Random', radius=5,
# 		font=pygame.font.SysFont('Verdana', 18, bold=True),
# 		onClick=lambda: pygame.event.post(pygame.event.Event(RANDOM_GAME_EVENT)),
# 		borderColor='black', borderThickness=2,
# 	)
# 	visualizer = Label(window, f'Visualize', 1055, 450, 16)
# 	toggle = Toggle(window, 1160, 455, 18, 22, handleRadius=11)
# 	bfs_button = Button(
# 		window, 1055, 280, 130, 40, text='Solve BFS', radius=5,
# 		font=pygame.font.SysFont('Verdana', 18, bold=True),
# 		onClick=lambda: pygame.event.post(pygame.event.Event(SOLVE_BFS_EVENT)),
# 		borderColor='black', borderThickness=2,
# 	)
# 	astarman_button = Button(
# 		window, 1055, 340, 130, 40, text='A* Manhattan', radius=5,
# 		font=pygame.font.SysFont('Verdana', 14, bold=True),
# 		onClick=lambda: pygame.event.post(pygame.event.Event(SOLVE_ASTARMAN_EVENT)),
# 		borderColor='black', borderThickness=2,
# 	)
# 	dijk_button = Button(
# 		window, 1055, 400, 130, 40, text='Dijkstra', radius=5,
# 		font=pygame.font.SysFont('Verdana', 14, bold=True),
# 		onClick=lambda: pygame.event.post(pygame.event.Event(SOLVE_DIJKSTRA_EVENT)),
# 		borderColor='black', borderThickness=2,
# 	)
# 	seed = Label(window, f'Seed', 1055, 190, 16)
# 	seedbox = TextBox(
# 		window, 1110, 191, 75, 28, placeholderText='Seed',
# 		borderColour=(0, 0, 0), textColour=(0, 0, 0),
# 		onSubmit=lambda: pygame.event.post(pygame.event.Event(RANDOM_GAME_EVENT)),
# 		borderThickness=1, radius=2,
# 		font=pygame.font.SysFont('Verdana', 14),
# 	)
# 	moves = Label(window, f' Moves - 0 ', 1055, 75, 20)
# 	paths = MultilineLabel(window, f'Solution Depth: 0\n', 64, 0, 20)
# 	level_clear = LevelClear(window, f'Level Clear!')
# 	return {
# 		'restart': restart,
# 		'random_button': random_game,
# 		'moves_label': moves,
# 		'prev_button': prev_button,
# 		'next_button': next_button,
# 		'label': label,
# 		'level_clear': level_clear,
# 		'toggle': toggle,
# 		'visualizer': visualizer,
# 		'bfs': bfs_button,
# 		'paths': paths,
# 		'seedbox': seedbox,
# 		'seed': seed,
# 		'astarman': astarman_button,
# 		'dijkstra': dijk_button,
# 	}


# class Label:
# 	def __init__(self, window, text, x, y, font_size, transparency=False, color='black'):
# 		self.x = x
# 		self.y = y
# 		self.font = pygame.font.SysFont('Verdana', font_size, bold=True)
# 		self.image = self.font.render(text, 1, color)
# 		self.max_width = self.image.get_width()
# 		self.total_height = self.image.get_height()
# 		self.rect = pygame.Rect(x, y, self.max_width + 10, self.total_height + 10)
# 		self.window = window
# 		self.transparency = transparency
# 		self.solved = False

# 	def set_text(self, new_text, font_size, color='black'):
# 		self.font = pygame.font.SysFont('Verdana', font_size, bold=True)
# 		self.image = self.font.render(new_text, 1, color)
# 		self.draw()

# 	def set_moves(self, new_text, font_size, color='black'):
# 		self.font = pygame.font.SysFont('Verdana', font_size, bold=True)
# 		self.image = self.font.render(new_text, 1, color)
# 		_, _, w, h = self.image.get_rect()
# 		self.rect.width = max(130, w)
# 		self.rect.height = max(40, h)
# 		self.draw()

# 	def draw(self):
# 		pygame.draw.rect(
# 			self.window,
# 			pygame.Color('gray'),
# 			(self.rect.x,
#     		self.rect.y,
# 			self.rect.width,
# 			self.rect.height)
# 		)
# 		pygame.draw.rect(
# 			self.window,
# 			(0, 0, 0),
# 			(self.rect.x,
#     		self.rect.y,
# 			self.rect.width,
# 			self.rect.height),
# 			width=3
# 		)
# 		text_pos_x = (self.rect.width - self.image.get_width()) // 2 + self.rect.x
# 		text_pos_y = (self.rect.height - self.image.get_height()) // 2 + self.rect.y
# 		self.window.blit(self.image, (text_pos_x, text_pos_y))


# class MultilineLabel(Label):
# 	def __init__(self, window, text, x, y, font_size, transparency=False, color='black'):
# 		super().__init__(window, text, x, y, font_size, transparency)
# 		self.lines = text.split('\n')
# 		if len(self.lines) == 1:
# 			self.image = self.font.render(text, 1, color)
# 		self.images = [self.font.render(line, 1, color) for line in self.lines]
# 		self.max_width = max(image.get_width() for image in self.images)
# 		self.total_height = (sum(image.get_height() for image in self.images) +
# 		  				(len(self.lines) - 1) * font_size // 2)
# 		self.rect = pygame.Rect(x, y, self.max_width + 10, self.total_height + 10)
# 		self.max_lines = len(self.lines)

# 	def reset(self, text=''):
# 		self.max_width = self.total_height = 1
# 		self.transparency = False
# 		self.solved = False
# 		self.max_lines = 2
# 		self.set_text(f'{text}\n', 20)
# 		pygame.display.update()

# 	def set_text(self, new_text, font_size, color='black'):
# 		self.font = pygame.font.SysFont('Verdana', font_size, bold=True)
# 		self.new_lines = new_text.split('\n')
# 		path_split = []
# 		for i in range(0, len(self.new_lines[1]), 60):
# 			path_split.append(self.new_lines[1][i:i + 60])
# 		self.lines = [self.new_lines[0]] + path_split
# 		self.max_lines = max(self.max_lines, len(self.lines))
# 		while len(self.lines) < self.max_lines:
# 			self.lines.append('')
# 		self.images = [self.font.render(line, 1, color) for line in self.lines]
# 		self.max_width = max(self.max_width, max(image.get_width() for image in self.images))
# 		self.total_height = (sum(image.get_height() for image in self.images) +
# 		  					 (len(self.lines) - 1) * font_size // 2)
# 		self.rect = pygame.Rect(self.x, self.y, self.max_width + 10, self.total_height + 10)
# 		self.draw()

# 	def draw(self):
# 		transparent_surface = pygame.Surface(
# 			(self.rect.width, self.rect.height), pygame.SRCALPHA
# 		)
# 		transparent_surface.set_alpha(110)
# 		transparent_surface.fill((200, 0, 0) if not self.solved else (0, 255, 0))
# 		if not self.transparency:
# 			pygame.draw.rect(
# 				self.window,
# 				(200, 0, 0) if not self.solved else (0, 255, 0),
# 				(self.rect.x,
# 				self.rect.y,
# 				self.rect.width,
# 				self.rect.height)
# 			)
# 		pygame.draw.rect(
# 			self.window,
# 			(0, 0, 0),
# 			(self.rect.x,
# 			self.rect.y,
# 			self.rect.width,
# 			self.rect.height),
# 			width=3
# 		)
# 		offset = ((self.rect.height - sum(image.get_height() for image in self.images) -
# 				   (len(self.images) - 1)) // 2 + self.rect.y)
# 		if self.transparency:
# 			self.window.blit(transparent_surface, (self.rect.x, self.rect.y))
# 		for image in self.images:
# 			text_pos_x = (self.rect.width - image.get_width()) // 2 + self.rect.x
# 			text_pos_y = offset
# 			offset += image.get_height()
# 			self.window.blit(image, (text_pos_x,text_pos_y))


# class LevelClear(Label):
# 	def __init__(self, window, text, x=256, y=192, font_size=60, color='black'):
# 		super().__init__(window, text, x, y, font_size, color)
# 		self.w, self.h = 512, 256
# 		self.rect = pygame.Rect(x, y, self.w, self.h)
# 		self.image = self.font.render(text, 1, color)

# 	def draw(self):
# 		transparent_surface = pygame.Surface(
# 			(self.rect.width, self.rect.height), pygame.SRCALPHA
# 		)
# 		transparent_surface.set_alpha(100)
# 		transparent_surface.fill((0, 255, 0))
# 		pygame.draw.rect(
# 			self.window,
# 			'#008000',
# 			(self.rect.x,
# 			self.rect.y,
# 			self.rect.width,
# 			self.rect.height),
# 			width=4
# 		)
# 		text_pos_x = (self.rect.width - self.image.get_width()) // 2 + self.rect.x
# 		text_pos_y = (self.rect.height - self.image.get_height()) // 2 + self.rect.y
# 		self.window.blit(transparent_surface, (self.rect.x, self.rect.y))
# 		self.window.blit(self.image, (text_pos_x,text_pos_y))
