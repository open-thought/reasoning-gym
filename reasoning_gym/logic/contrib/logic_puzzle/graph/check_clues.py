from z3 import *
import solver as solver
import itertools
from collections import defaultdict
import sat_utils

def check_singe_clues(puzzle_idx, answer_header, answer_value, running_clues, idx2clue, used_clue, all_cells):
    solved_cell = defaultdict(int)
    valid_solutions = defaultdict(int)
    print('check single clues')
    for idx in idx2clue:
        if idx not in used_clue:
            new_clues = running_clues+idx2clue[idx].as_cnf()
            numbered_cnf, num2var = sat_utils.translate(new_clues)
            single_solver = solver.my_solver(puzzle_idx, answer_header, answer_value, numbered_cnf, num2var)
            cell_info = single_solver.check_cell_difficulty()
            num_cell=0
            for cell in cell_info:
                if cell not in all_cells:
                    num_cell+=1
            solved_cell[idx] = num_cell
            del (single_solver)
    idx = sorted(solved_cell.items(),key=lambda x: x[1], reverse=True)[0][0]
    if solved_cell[idx]!=0:
        return [idx]
    else:
        return [1000]

def check(puzzle_idx, answer_header, answer_value, running_clues, idx2clue, used_clue, all_cells):
    idx = check_singe_clues(puzzle_idx, answer_header, answer_value, running_clues, idx2clue, used_clue, all_cells)
    if idx[0]<1000:
        return idx
    else:
        print('check multiple clues')
        solved_cell = defaultdict(int)
        all_left_clues_idx = [i for i in idx2clue if i not in used_clue]
        print(all_left_clues_idx)
        for n in range(2, 10):
            combinations = itertools.combinations(all_left_clues_idx, n)
            for comb in combinations:
                new_clues = [clues for clues in running_clues]
                for comb_idx in comb:
                     new_clues += idx2clue[comb_idx].as_cnf()
                numbered_cnf, num2var = sat_utils.translate(new_clues)
                single_solver = solver.my_solver(puzzle_idx, answer_header, answer_value, numbered_cnf, num2var)
                cell_info = single_solver.check_cell_difficulty()
                num_cell = 0
                for cell in cell_info:
                    if cell not in all_cells:
                        num_cell += 1
                solved_cell[comb] = num_cell
                del (single_solver)
                if num_cell!=0:
                    return comb
