import random
import time

import pygame
import pygame_widgets

from src.astar import solve_astar
from src.bfs import solve_bfs
from src.events import *
from src.game import Game
from src.generator import generate
from src.utils import play_solution
from src.widgets import sidebar_widgets

random.seed(6)


def play_game(window, level=1, random_game=False, random_seed=None, **widgets):
	moves = runtime = 0
	show_solution = False
	widgets['paths'].transparency = False
	if random_game:
		if not random_seed:
			random_seed = random.randint(0, 99999)
		generate(window, seed=random_seed, visualizer=widgets['toggle'].getValue())
	if level <= 1:
		widgets['prev_button'].hide()
	else:
		widgets['prev_button'].show()
	if level >= 7:
		widgets['next_button'].hide()
	else:
		widgets['next_button'].show()
	if random_game or level == 0:
		widgets['label'].set_text(f'Seed {random_seed}', 18)
	else:
		widgets['label'].set_text(f'Level {level}', 30)
	game = Game(level=level, window=window)
	game_loop = True
	while game_loop:
		events = pygame.event.get()
		for event in events:
			if event.type == pygame.QUIT:
				game_loop = False
				return {
					'keep_playing': False,
					'reset': -1, 
					'random_game': False,
				}
			elif event.type == RESTART_EVENT:
				game_loop = False
				print(f'Restarting level {level}\n')
				window.fill((0, 0, 0, 0))
				return {
					'keep_playing': True,
					'reset': level, 
					'random_game': random_game,
					'random_seed': random_seed,
				}
			elif event.type == PREVIOUS_EVENT:
				game_loop = False
				print(f'Previous level {level - 1}\n')
				window.fill((0, 0, 0, 0))
				return {
					'keep_playing': True,
					'reset': level - 1, 
					'random_game': False
				}
			elif event.type == NEXT_EVENT:
				game_loop = False
				print(f'Next level {level + 1}\n')
				window.fill((0, 0, 0, 0))
				return {
					'keep_playing': True,
					'reset': level + 1, 
					'random_game': False
				}
			elif event.type == RANDOM_GAME_EVENT:
				game_loop = False
				print('Loading a random puzzle\n')
				window.fill((0, 0, 0, 0))
				new_seed = None
				try:
					new_seed = int(widgets['seedbox'].getText())
					if new_seed < 1 or new_seed > 99999:
						new_seed = None
						raise ValueError('Seed must be between 1 and 99999')
				except ValueError as e:
					print(e)
				return {
					'keep_playing': True,
					'reset': 0, 
					'random_game': True,
					'random_seed': new_seed
				}
			elif event.type == SOLVE_BFS_EVENT:
				print('Finding a solution for the puzzle\n')
				widgets['paths'].reset('Solving with [BFS]')
				show_solution = True
				start = time.time()
				solution, depth = solve_bfs(
					game.get_matrix(), 
					widget=widgets['paths'], 
					visualizer=widgets['toggle'].getValue()
				)
				runtime = round(time.time() - start, 5)
				if solution:
					widgets['paths'].solved = True
					widgets['paths'].transparency = True
					widgets['paths'].set_text(
						f'[BFS] Solution Found in {runtime}s!\n{solution}',
						20,
					)
					moves = play_solution(solution, game, widgets, show_solution, moves)
				else:
					widgets['paths'].solved = False
					widgets['paths'].set_text(
						'[BFS] Solution Not Found!\n' + 
						('Deadlock Found!' if depth < 0 else f'Depth {depth}'), 
						20,
					)
			elif event.type == SOLVE_ASTARMAN_EVENT:
				print('Finding a solution for the puzzle\n')
				widgets['paths'].reset('Solving with [A*]')
				show_solution = True
				start = time.time()
				solution, depth = solve_astar(
					game.get_matrix(), 
					widget=widgets['paths'], 
					visualizer=widgets['toggle'].getValue(),
					heuristic='manhattan',
				)
				runtime = round(time.time() - start, 5)
				if solution:
					widgets['paths'].solved = True
					widgets['paths'].transparency = True
					widgets['paths'].set_text(
						f'[A*] Solution Found in {runtime}s!\n{solution}',
						20,
					)
					moves = play_solution(solution, game, widgets, show_solution, moves)
				else:
					widgets['paths'].solved = False
					widgets['paths'].set_text(
						'[A*] Solution Not Found!\n' + 
						('Deadlock Found!' if depth < 0 else f'Depth {depth}'), 
						20,
					)
			elif event.type == SOLVE_DIJKSTRA_EVENT:
				print('Finding a solution for the puzzle\n')
				widgets['paths'].reset('Solving with [Dijkstra]')
				show_solution = True
				start = time.time()
				solution, depth = solve_astar(
					game.get_matrix(), 
					widget=widgets['paths'], 
					visualizer=widgets['toggle'].getValue(),
					heuristic='dijkstra',
				)
				runtime = round(time.time() - start, 5)
				if solution:
					widgets['paths'].solved = True
					widgets['paths'].transparency = True
					widgets['paths'].set_text(
						f'[Dijkstra] Solution Found in {runtime}s!\n{solution}',
						20
					)
					moves = play_solution(solution, game, widgets, show_solution, moves)
				else:
					widgets['paths'].solved = False
					widgets['paths'].set_text(
						'[Dijkstra] Solution Not Found!\n' + 
						('Deadlock Found!' if depth < 0 else f'Depth {depth}'), 
						20,
					)
			elif event.type == pygame.KEYDOWN:
				if event.key in (pygame.K_d, pygame.K_RIGHT):
					moves += game.player.update(key='R')
				elif event.key in (pygame.K_a, pygame.K_LEFT):
					moves += game.player.update(key='L')
				elif event.key in (pygame.K_w, pygame.K_UP):
					moves += game.player.update(key='U')
				elif event.key in (pygame.K_s, pygame.K_DOWN):
					moves += game.player.update(key='D')
		game.floor_group.draw(window)
		game.goal_group.draw(window)
		game.object_group.draw(window)
		pygame_widgets.update(events)
		widgets['label'].draw()
		widgets['seed'].draw()
		widgets['visualizer'].draw()
		widgets['moves_label'].set_moves(f' Moves - {moves} ', 20)
		if show_solution:
			widgets['paths'].draw()
		pygame.display.update()
		if game.is_level_complete():
			print(f'Level Complete! - {moves} moves')
			widgets['level_clear'].draw()
			pygame.display.update()
			game_loop = False
			wait = True 
			while wait:
				for event in pygame.event.get():
					if event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
						wait = False
	del game
	print('Objects cleared!\n')
	return {
		'keep_playing': True,
		'reset': 0 if random_game else -1, 
		'random_game': random_game,
	}


def main():
	pygame.init()
	displayIcon = pygame.image.load('img/icon.png')
	pygame.display.set_icon(displayIcon)
	window = pygame.display.set_mode((1216, 640))
	pygame.display.set_caption('Sokoban')
	level = 1
	keep_playing = True
	random_game = False
	random_seed = None
	widgets = sidebar_widgets(window)
	while keep_playing:
		print(f'Loading level {level}\n' if level > 0 else 'Loading random game')
		game_data = play_game(window, level, random_game, random_seed, **widgets)
		keep_playing = game_data.get('keep_playing', False)
		if not keep_playing:
			pygame.quit()
			quit()
		reset = game_data.get('reset', -1)
		random_game = game_data.get('random_game', False)
		random_seed = game_data.get('random_seed')
		level = reset if reset >= 0 else min(level + 1, 7)

	
if __name__ == '__main__':
	# wall: +, box: @, player: *, goal: X, box on goal: $, player on goal: %, empty: -
	main()