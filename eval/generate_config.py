#!/usr/bin/env python
"""
Configuration generator for reasoning-gym evaluation.

This script generates a YAML configuration file with all registered datasets
from reasoning_gym, organized by category.

Usage:
    python generate_config.py [options]

Options:
    --output OUTPUT       Output YAML file path (default: all_datasets.yaml)
    --model MODEL         Model name (default: openai/gpt-4)
    --provider PROVIDER   Provider name (default: None)
    --size SIZE           Default dataset size (default: 100)
    --seed SEED           Default dataset seed (default: 42)
    --include-params      Include all configuration parameters (default: False)
"""

import argparse
import inspect
from collections import defaultdict
from dataclasses import fields

import yaml

from reasoning_gym.factory import DATASETS


def extract_category(module_name):
    """Extract category from module name."""
    parts = module_name.split(".")
    if len(parts) >= 3:
        return parts[1]  # reasoning_gym.{category}.dataset_name
    return "other"


def generate_config(model, provider, size, seed, include_params):
    """Generate configuration with all registered datasets."""
    # Group datasets by category
    categories = defaultdict(list)

    for dataset_name, (dataset_cls, config_cls) in DATASETS.items():
        # Extract category from module name
        category = extract_category(dataset_cls.__module__)

        # Create dataset entry
        dataset_entry = {"dataset": dataset_name}

        # Optionally include all configuration parameters
        if include_params:
            params = {}
            # Get default values from config class fields
            for field in fields(config_cls):
                # Skip seed and size as they're handled separately
                if field.name not in ["seed", "size"]:
                    # Only include fields with default values
                    if field.default != inspect.Parameter.empty:
                        params[field.name] = field.default

            if params:
                dataset_entry["params"] = params

        # Add to appropriate category
        categories[category].append(dataset_entry)

    # Create configuration structure
    config = {
        "model": model,
        "provider": provider,
        "output_dir": "results",
        "max_concurrent": 10,
        "default_size": size,
        "default_seed": seed,
        "categories": [],
    }

    # Add categories
    for category_name, datasets in sorted(categories.items()):
        config["categories"].append({"category": category_name, "datasets": datasets})

    return config


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation configuration with all datasets")
    parser.add_argument("--output", default="all_datasets.yaml", help="Output YAML file path")
    parser.add_argument("--model", default="openai/gpt-4", help="Model name")
    parser.add_argument("--provider", default=None, help="Provider name")
    parser.add_argument("--size", type=int, default=100, help="Default dataset size")
    parser.add_argument("--seed", type=int, default=42, help="Default dataset seed")
    parser.add_argument("--include-params", action="store_true", help="Include all configuration parameters")

    args = parser.parse_args()

    # Generate configuration
    config = generate_config(
        model=args.model, provider=args.provider, size=args.size, seed=args.seed, include_params=args.include_params
    )

    # Write to file
    with open(args.output, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(
        f"Configuration with {sum(len(cat['datasets']) for cat in config['categories'])} datasets written to {args.output}"
    )
    print(f"Categories: {', '.join(cat['category'] for cat in config['categories'])}")


if __name__ == "__main__":
    main()
