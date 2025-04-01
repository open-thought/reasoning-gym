#!/usr/bin/env -S PYTHONHASHSEED=1 python3

import argparse
import csv
import os
import sys
import textwrap
from pathlib import Path

from tqdm import tqdm

import reasoning_gym.code.bf
from reasoning_gym.factory import CURRICULA, DATASETS, create_curriculum, create_dataset


def main():
    """Evaluate a seeded benchmark and save it to a CSV"""
    parser = argparse.ArgumentParser(description="Evaluate a benchmark result.")
    parser.add_argument("--input", type=str, default="input.csv", help="Input filename")
    parser.add_argument("--num-per-dataset", type=int, default=10, help="Number of items per dataset (default: 10).")
    args = parser.parse_args()

    seed_str = os.getenv("RNG_SEED")
    if seed_str is None:
        raise EnvironmentError("The 'RNG_SEED' environment variable is not set.")

    try:
        seed = int(seed_str)
    except ValueError:
        raise ValueError(f"Invalid seed: '{seed_str}' is not an integer.")

    # Increase CSV field size limit to avoid errors with large fields.
    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = int(max_int / 10)

    results = {}
    with open(args.input, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:

            name = row["source_dataset"]
            dataset_cls, config_cls = DATASETS[name]

            curriculum = create_curriculum(name)

            # Get dataset class docstring if available
            dataset = create_dataset(name, seed=seed)

            # Find the maximum number of levels across all attributes
            max_level = max(len(attr.levels) for attr in curriculum.attributes.values()) - 1

            curriculum = create_curriculum(name)
            # Set all attributes to this max using the global level function
            curriculum.set_global_level(max_level)
            # Generate config with this level
            config = curriculum.generate_configuration({"seed": seed, "size": args.num_per_dataset})

            dataset = dataset_cls(config=config)
            for item in dataset:
                score = dataset.score_answer(answer=row["answer"], entry=item)
                results.setdefault(name, []).append(score)

    # Compute individual percentage scores per dataset and overall total percentage
    individual_scores = {}
    all_scores = []

    for name, scores in results.items():
        if scores:
            avg_score = sum(scores) / len(scores)
        else:
            avg_score = 0.0
        individual_scores[name] = avg_score * 100  # convert to percentage
        all_scores.extend(scores)

    # Compute the total score over all items
    if all_scores:
        total_score = (sum(all_scores) / len(all_scores)) * 100  # percentage
    else:
        total_score = 0.0

    final_results = {
        "individual_scores": individual_scores,
        "total_score": total_score,
    }

    print(final_results)


if __name__ == "__main__":
    main()
