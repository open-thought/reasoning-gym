import pickle
from tqdm import tqdm
from z3 import *
import json
import argparse
import solver as solver
import check_clues
import graph_literals
from collections import defaultdict
import sat_utils


def get_idx2clue(clues):
    clue_num2_clue = defaultdict(int)
    clue_type2_clue_num = defaultdict(list)
    clue_num = 0
    for clue in list(clues):
        clue_num2_clue[clue_num] = clue
        clue_type2_clue_num[type(clue)].append(clue_num)
        clue_num += 1
    return clue_num2_clue, clue_type2_clue_num

def logic_grid_puzzle(inputfile, ground_truth, size, lower_part, higher_part):
    reasoning_result = []
    answers = json.load(open(ground_truth, 'r'))
    puzzles = pickle.load(open(inputfile, 'rb'))
    cell_difficulty = {}
    mode = inputfile[inputfile.find('puzzles.') + 8:inputfile.find('.pkl')]
    print('Number of puzzles', len(answers))
    assert len(answers) == len(puzzles)
    new_data = []
    for i in tqdm(range(len(puzzles[:]))):
        d = puzzles[i]
        per_size_idx = int(puzzles[i]['idx'].split('-')[-1])
        if d['idx'].startswith("lgp-"+mode+"-"+size) and per_size_idx>=lower_part and per_size_idx<higher_part:
            print('Puzzle id:', d['idx'])
            print('Puzzle', d)
            print("Solving puzzle"+"==============="*7+"Solving puzzle")

            assert d['idx']==answers[i]['idx']
            answer_header = answers[i]['solution']['table_header']
            answer_value = answers[i]['solution']['table_rows']

            puzzle_clues = d['puzzle'].clues
            idx2clue, clue_type2_idx = get_idx2clue(puzzle_clues)

            running_clues = []
            all_cells = []
            used_clue = []
            self_constraints = d['puzzle'].constraints
            running_clues.extend(self_constraints)

            # for i in range(len(idx2clue)):
            first=True
            step_num=1
            reasoning = ""
            while len(used_clue)<len(idx2clue):
                reasoning += 'Step {}: '.format(step_num)
                step_num+=1
                clue_idxs = check_clues.check(d['idx'],
                                              answer_header, answer_value,
                                              running_clues, idx2clue, used_clue, all_cells)
                for current_clue_idx in clue_idxs:
                    running_clues.extend(idx2clue[current_clue_idx].as_cnf())
                    used_clue.append(current_clue_idx)

                numbered_cnf, num2var = sat_utils.translate(running_clues)
                single_solver = solver.my_solver(d['idx'], answer_header, answer_value, numbered_cnf, num2var)
                cell_info = single_solver.check_cell_difficulty()
                new_cell=[]
                for cell in cell_info:
                    if cell not in all_cells:
                        new_cell.append(cell)
                        all_cells.append(cell)
                reasoning += graph_literals.print_clue(clue_idxs, idx2clue, new_cell, first)
                first=False
                reasoning+='\n'
            assert len(all_cells) == len(answer_value)*(len(answer_value[0])-1)
            reasoning+='The puzzle is solved.'
            reasoning_result.append(reasoning)
            print(reasoning)
            single_data = answers[i]
            single_data["reasoning"]=reasoning
            new_data.append(single_data)
    
    with open("./data/logic_grid_puzzles.reasoning."+mode+size+"-"+str(lower_part)+"_"+str(higher_part)+".json", "w") as outputfile:
        json.dump(new_data, outputfile)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str, default="./data/logic_grid_puzzles.test_id_xl.pkl")
    parser.add_argument('--ground_truth', type=str, default="./data/ogic_grid_puzzles.test_id_xl.json")
    parser.add_argument('--size', type=str, default="2x")
    parser.add_argument('--lower_part', type=int, default=0) #min data index
    parser.add_argument('--higher_part', type=int, default=100) #max data index
    args = parser.parse_args()

    logic_grid_puzzle(args.input_data, args.ground_truth, args.size, args.lower_part, args.higher_part)
    return 1

if __name__ == "__main__":
    main()
