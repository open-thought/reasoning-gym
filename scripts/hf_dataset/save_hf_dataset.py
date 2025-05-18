#!/usr/bin/env python3
"""
Script to generate Reasoning Gym datasets and save them to the Hugging Face Hub.
"""

import argparse
from typing import Dict, List, Optional

import yaml
from datasets import Dataset
from tqdm import tqdm

from reasoning_gym.composite import DatasetSpec
from reasoning_gym.factory import DATASETS, create_dataset


def generate_dataset(
    dataset_names: List[str],
    dataset_size: int = 20000,
    seed: int = 42,
    weights: Optional[Dict[str, float]] = None,
    configs: Optional[Dict[str, Dict]] = None,
) -> Dataset:
    """
    Generate a dataset from the specified Reasoning Gym datasets.

    Args:
        dataset_names: List of dataset names to include
        dataset_size: Total size of the dataset to generate
        seed: Random seed for dataset generation
        weights: Optional dictionary mapping dataset names to weights
        configs: Optional dictionary mapping dataset names to configurations

    Returns:
        A Hugging Face Dataset object
    """
    # Validate dataset names
    for name in dataset_names:
        if name not in DATASETS:
            raise ValueError(f"Dataset '{name}' not found. Available datasets: {sorted(DATASETS.keys())}")

    # Set default weights if not provided
    if weights is None:
        equal_weight = 1.0 / len(dataset_names)
        weights = {name: equal_weight for name in dataset_names}
    else:
        # Validate weights
        for name in dataset_names:
            if name not in weights:
                weights[name] = 0.0
                print(f"Warning: No weight provided for {name}, setting to 0.0")

    # Set default configs if not provided
    if configs is None:
        configs = {name: {} for name in dataset_names}
    else:
        # Add empty configs for missing datasets
        for name in dataset_names:
            if name not in configs:
                configs[name] = {}

    # Create dataset specs
    dataset_specs = [DatasetSpec(name=name, weight=weights[name], config=configs[name]) for name in dataset_names]

    # Create composite dataset
    data_source = create_dataset("composite", seed=seed, size=dataset_size, datasets=dataset_specs)

    # Generate all examples
    examples = []
    for idx in tqdm(range(dataset_size), desc="Generating examples"):
        example = data_source[idx]
        examples.append(example)

    # Convert to HF Dataset
    hf_dataset = Dataset.from_list(examples)
    return hf_dataset


def save_to_hub(
    dataset: Dataset,
    repo_id: str,
    token: Optional[str] = None,
    private: bool = False,
    commit_message: str = "Upload reasoning_gym dataset",
    split: Optional[str] = None,
) -> str:
    """
    Save the dataset to the Hugging Face Hub.

    Args:
        dataset: HF Dataset to save
        repo_id: Hugging Face repo ID (e.g., "username/dataset-name")
        token: HF API token
        private: Whether the repository should be private
        commit_message: Commit message
        split: Dataset split name

    Returns:
        URL of the uploaded dataset
    """
    # Push to the hub
    dataset.push_to_hub(
        repo_id,
        token=token,
        private=private,
        commit_message=commit_message,
    )

    print(f"Dataset pushed to https://huggingface.co/datasets/{repo_id}")
    return f"https://huggingface.co/datasets/{repo_id}"


def load_config(config_path: str) -> dict:
    """
    Load dataset configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Dictionary containing the configuration
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def main():
    parser = argparse.ArgumentParser(description="Generate and upload Reasoning Gym datasets to HF Hub")
    parser.add_argument("--dataset", type=str, required=False, help="Dataset names (comma-separated list)")
    parser.add_argument("--config", type=str, required=False, help="Path to dataset configuration YAML file")
    parser.add_argument("--size", type=int, default=20000, help="Total dataset size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--repo-id", type=str, help="Hugging Face repository ID (e.g., 'username/dataset-name')")
    parser.add_argument("--private", action="store_true", help="Make the HF repository private")
    parser.add_argument(
        "--split", type=str, choices=["train", "test", "validation"], default="train", help="Dataset split name"
    )

    # First parse args to check for config file
    args, unknown = parser.parse_known_args()

    # If config specified, load it to handle repo_id
    repo_id_from_config = None
    if args.config:
        config = load_config(args.config)
        if "huggingface" in config and "repo_id" in config["huggingface"]:
            repo_id_from_config = config["huggingface"]["repo_id"]

    # Re-parse with defaults if needed
    if repo_id_from_config:
        parser.set_defaults(repo_id=repo_id_from_config)

    args = parser.parse_args()

    # Validate repo_id is provided
    if not args.repo_id:
        parser.error(
            "--repo-id is required. Provide it via command line or in the config file under huggingface.repo_id"
        )

    # Load configuration
    dataset_names = []
    weights = {}
    configs = {}

    # Load from config file if provided
    if args.config:
        config = load_config(args.config)
        if "reasoning_gym" in config:
            rg_config = config["reasoning_gym"]
            if "datasets" in rg_config:
                for name, ds_config in rg_config["datasets"].items():
                    dataset_names.append(name)
                    weights[name] = ds_config.get("weight", 1.0 / len(rg_config["datasets"]))
                    configs[name] = ds_config.get("config", {})

            # Get dataset size from config if not explicitly provided
            if "dataset_size" in rg_config and args.size == 20000:  # Only use if default size
                args.size = rg_config["dataset_size"]

        # Check for HF settings in config
        if "huggingface" in config:
            hf_config = config["huggingface"]
            if "private" in hf_config:
                args.private = hf_config["private"]
            if "split" in hf_config and args.split == "train":  # Only override if using default
                args.split = hf_config["split"]

    # Override with command line arguments if provided
    if args.dataset:
        dataset_names = [name.strip() for name in args.dataset.split(",")]
        # Reset weights if datasets are provided
        equal_weight = 1.0 / len(dataset_names)
        weights = {name: equal_weight for name in dataset_names}

    # Validate that we have dataset names
    if not dataset_names:
        parser.error("No datasets specified. Use --dataset or --config to specify datasets.")

    print(f"Generating dataset with {len(dataset_names)} datasets: {', '.join(dataset_names)}")
    print(f"Dataset size: {args.size}")
    print(f"Dataset seed: {args.seed}")
    print(f"Repository ID: {args.repo_id}")

    # Generate the dataset
    dataset = generate_dataset(
        dataset_names=dataset_names,
        dataset_size=args.size,
        seed=args.seed,
        weights=weights,
        configs=configs,
    )

    # Save to hub with specified split
    save_to_hub(
        dataset=dataset,
        repo_id=args.repo_id,
        private=args.private,
        commit_message=f"Upload reasoning_gym dataset with {len(dataset_names)} datasets: {', '.join(dataset_names)}",
        split=args.split,
    )

    print("Done!")


if __name__ == "__main__":
    # Example usage:
    # python save_hf_dataset.py --config example_hf_dataset_config.yaml
    # python save_hf_dataset.py --dataset "number_sorting,spell_backward,word_sorting" --repo-id "username/reasoning-gym-dataset"
    main()
