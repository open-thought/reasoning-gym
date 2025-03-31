#!/usr/bin/env -S PYTHONHASHSEED=1 python3

import argparse
import csv
import os
import textwrap
from pathlib import Path

from tqdm import tqdm

import reasoning_gym.code.bf
from reasoning_gym.factory import CURRICULA, DATASETS, create_curriculum, create_dataset


def generate_questions(seed: int = 123, num: int = 1) -> str:
    """Generate questions for all curricula at max difficulty"""

    datasets_with_curricula = sorted([name for name in DATASETS.keys() if name in CURRICULA])
    datasets_without_curricula = sorted([name for name in DATASETS.keys() if name not in CURRICULA])

    rows = []
    for name in datasets_with_curricula:
        try:
            # Get curriculum and dataset class
            dataset_cls, config_cls = DATASETS[name]
            print(str(dataset_cls))
            curriculum = create_curriculum(name)

            # Get dataset class docstring if available
            dataset = create_dataset(name, seed=seed)

            # Find the maximum number of levels across all attributes
            max_level = max(len(attr.levels) for attr in curriculum.attributes.values()) - 1

            curriculum = create_curriculum(name)
            # Set all attributes to this max using the global level function
            curriculum.set_global_level(max_level)
            # Generate config with this level
            config = curriculum.generate_configuration({"seed": seed, "size": num})

            # Create dataset with this config
            dataset = dataset_cls(config=config)
            for item in dataset:
                row = {
                    "_rgid": item.get("metadata", {}).get("_rgid", ""),
                    "_rgidx": item.get("metadata", {}).get("_rgidx", ""),
                    "source_dataset": item.get("metadata", {}).get("source_dataset", ""),
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                }
                rows.append(row)
        except Exception as e:
            print(e)

    return rows


def main():
    """Create a seeded benchmark and save it to a CSV"""
    parser = argparse.ArgumentParser(description="Generate a seeded benchmark.")
    parser.add_argument("--num-per-dataset", type=int, default=10, help="Number of items per dataset (default: 10).")
    parser.add_argument("--output", type=str, default="output.csv", help="Output filename")
    parser.add_argument("--answers", action="store_true", help="Include answers in output CSV")
    args = parser.parse_args()

    seed_str = os.getenv("RNG_SEED")

    if seed_str is None:
        raise EnvironmentError("The 'RNG_SEED' environment variable is not set.")

    try:
        seed = int(seed_str)
    except ValueError:
        raise ValueError(f"Invalid seed: '{seed_str}' is not an integer.")

    rows = generate_questions(seed, num=args.num_per_dataset)

    fieldnames = ["_rgid", "_rgidx", "source_dataset", "question", "answer"]
    with open(args.output, "w", newline="", encoding="utf-8") as csvfile:
        # Define the header columns in the desired order
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header row
        writer.writeheader()

        for row in rows:
            if not args.answers: 
                row['answer'] = ''
            writer.writerow(row)


if __name__ == "__main__":
    main()
