import argparse
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Sequence
from tqdm import tqdm


def parse_table(table_string):
    flat_rows = [r for r in table_string.split('\n') if r.startswith('$')]
    table = {}
    for flat_row in flat_rows:
        flat_cells = [c for c in flat_row.split('|') if c]
        house, attributes = flat_cells[0], flat_cells[1:]
        h_prefix, h_number = house.split(':')
        assert h_prefix == '$ House'
        house_key = f'House {h_number.strip()}'
        table[house_key] = {}

        for att in attributes:
            if not ':' in att:
                continue
            name, value = att.split(':')
            table[house_key][name.strip()] = value.strip()

    return table


def copy_table(table):
    new_table = {}
    for house, attributes in table.items():
        new_table[house] = {}
        for att_name, att_value in attributes.items():
            new_table[house][att_name] = None
    return new_table


def parse_step(step_text, table_to_fill):
    parsed_clues = [s.split('>')[0] for s in step_text.split('<') if '>' in s]
    ans_texts = [s.strip() for s in step_text.split('We know that ')[-1].split('.') if s.strip().startswith('The ')]
    for ans_text in ans_texts:
        try:
            att_name = ans_text.split(' in house ')[0].split('The ')[-1].strip()
            house_num = ans_text.split(' in house ')[1].split(' is ')[0].strip()
            att_value = ans_text.split(' is ')[-1].strip()
            table_to_fill[f'House {house_num}'][att_name] = att_value
        except:
            print(parsed_clues)
            print(ans_text)
            print(table_to_fill)
            print()
            continue
    return parsed_clues, table_to_fill


def pre_process(src_file: os.PathLike, dest_dir: os.PathLike, overwrite: bool = False):
    outputs = json.load(open(src_file, 'r'))['data']

    src_file = Path(src_file)
    os.makedirs(dest_dir, exist_ok=True)
    save_file = os.path.join(dest_dir, f"parsed_{src_file.name}")

    if not overwrite and os.path.exists(save_file):
        return

    items = []
    for output in tqdm(outputs, desc=f"{src_file.stem}"):
        groundtruth_table = parse_table(output['truth_outputs'][0])
        if type(output['output_text']) == list:
            output_text = output['output_text'][0].split('\n')
        else:
            output_text = output['output_text'].split('\n')
        step_texts = [o for o in output_text if o.startswith('Step ')]
        steps = []
        partial_table = copy_table(groundtruth_table)
        for s_text in step_texts:
            clues, filled_table = parse_step(s_text, partial_table)
            steps.append({
                'clues': clues,
                'partial_table': filled_table,
            })
            partial_table = deepcopy(filled_table)

        predict_table_texts = [o for o in output_text if o.startswith('$ House')]
        if not predict_table_texts:
            predicted_table = partial_table
        else:
            predicted_table = parse_table('\n'.join(predict_table_texts))

        items.append({
            'groundtruth_table': groundtruth_table,
            'predicted_table': predicted_table,
            'steps': steps
        })

    with open(save_file, 'w') as f:
        json.dump(items, f, indent=4)


def error_analysis(home_path: os.PathLike, file_names: Sequence[str] = None):
    if not file_names:
        parsed_files = [p for p in Path(home_path).glob("parsed_table*judged.json")]
    else:
        parsed_files = [os.path.join(home_path, f) for f in file_names]

    datasets = []
    for parsed_file in parsed_files:
        datasets.extend([json.loads(s) for s in open(parsed_file, 'r').readlines()])

    error_types = ['correct', 'type1', 'type2', 'type3']
    error_stats = {}
    for data in tqdm(datasets):
        gt_table = data['groundtruth_table']
        previous_type = None
        for i, step in enumerate(data["steps"]):
            partial_table = step['partial_table']
            # whether value correct:
            value_to_check = []
            for house_name, attributes in partial_table.items():
                for att, value in attributes.items():
                    if not i:
                        previous_empty = True
                    else:
                        if att not in data["steps"][i-1]['partial_table'][house_name]:
                            previous_empty = False
                        else:
                            previous_empty = data["steps"][i-1]['partial_table'][house_name][att] is None
                    if value is not None and previous_empty:
                        value_to_check.append((house_name, att))

            value_correct = all(
                [partial_table[house_name][att] == gt_table[house_name][att] for house_name, att in value_to_check]
            )
            operation_correct = step['label']
            if not i:
                ancestor_correct = True
            else:
                ancestor_correct = previous_type == 'correct'

            if value_correct:
                if ancestor_correct:
                    node_type = 'correct'
                else:
                    node_type = 'type3'
            else:
                if operation_correct:
                    node_type = 'type2'
                else:
                    node_type = 'type1'

            previous_type = node_type

            if i not in error_stats.keys():
                error_stats[i] = {e: 0 for e in error_types}
            error_stats[i][node_type] += 1

    print(json.dumps(error_stats))
    percent_stats = {}
    number_nodes = {}
    for key, values in error_stats.items():
        percent_stats[key] = {}
        total_nodes = sum([values[t] for t in error_types])
        number_nodes[key] = total_nodes
        for t in error_types:
            percent_stats[key][t] = values[t] / total_nodes
    print(json.dumps(percent_stats))
    print(number_nodes)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cot_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='tmp')
    parser.add_argument('--overwrite', action='store_true', default=False)

    args = parser.parse_args()

    cot_dir = Path(args.cot_dir)
    assert cot_dir.exists(), "cot_dir does not exist"
    assert cot_dir.is_dir(), "cot_dir is not a directory"

    for cot_file in cot_dir.glob("table*.json"):
        pre_process(cot_file, args.output_dir, args.overwrite)

    error_analysis(args.output_dir)


if __name__ == '__main__':
    main()
