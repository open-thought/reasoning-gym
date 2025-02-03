import pickle
from tqdm import tqdm
from z3 import *
import json
import argparse
import solver
import sat_utils

def solve_logic_grid_puzzle(inputfile, ground_truth):
    answers = [json.loads(answer) for answer in list(open(ground_truth, 'r'))][0]
    cell_difficulty = {}
    with open(inputfile, 'rb') as f:
        puzzles = pickle.load(f)
        for i in tqdm(range(len(puzzles[:]))):
            d=puzzles[i]
            assert d['idx']==answers[i]['idx']
            answer_header = answers[i]['solution']['table_header']
            answer_value = answers[i]['solution']['table_rows']
            # read cnf form
            symbolic_cnf = d['puzzle'].as_cnf()
            numbered_cnf, num2var = sat_utils.translate(symbolic_cnf)

            single_solver = solver.my_solver(d['idx'], answer_header, answer_value, numbered_cnf, num2var)
            solution, unique = single_solver.check_solution()
            if unique:
                for row_num in range(len(solution)):
                    for column_num in range(len(solution[row_num])):
                        assert row_num == solution[row_num][column_num]
            difficulty = single_solver.check_problem_difficulty()
            cell_difficulty[d['idx']]=difficulty

    with open('./logic_grid_puzzles.test.difficulty.pkl', 'wb') as outputfile:
        pickle.dump(cell_difficulty, outputfile)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str, default="../logic_grid_puzzles.test.pkl")
    parser.add_argument('--ground_truth', type=str, default="../logic_grid_puzzles.test.json")
    args = parser.parse_args()

    solve_logic_grid_puzzle(args.input_data, args.ground_truth)
    return 1

if __name__ == "__main__":
    main()
