#!/usr/bin/env python3
"""
main.py – Orchestrates the overall flow:
1. Generate word ladder sets
2. Submit chain-of-thought reasoning requests in batches via the LLM
3. Upload the final dataset to HuggingFace Hub (if needed)
"""

import sys
import uuid
from pathlib import Path
from typing import Any

from examples.word_ladder.utils import create_word_ladders, generate_reasoning


def create_dataset(jsonl_path: Path, config: dict[str, Any]) -> bool:
    """
    Creates the word ladder dataset, handling potential exhaustion gracefully.

    Returns:
        bool: True if dataset was created (even if truncated), False if creation failed
    """
    try:
        print("Step 1: Algorithmically creating word ladder chains...")
        create_word_ladders.create_word_ladder_dataset(str(jsonl_path), config=config)
        return True

    except IndexError as e:
        # Dataset was exhausted but some examples were generated
        print("\nNote: Dataset generation stopped early due to exhaustion of unique puzzles.")
        print(f"Reason: {str(e)}")
        if jsonl_path.exists():
            print("Continuing with the partial dataset that was successfully generated.")
            return True
        return False

    except Exception as e:
        # Unexpected error during dataset creation
        print(f"\nError: Failed to create dataset: {str(e)}")
        return False


def main():
    # Centralized configuration for the dataset
    config = {
        "dataset_name": "word_ladder",
        "dataset_config": {
            "min_word_length": 3,
            "max_word_length": 3,
            "min_chain_length": -1,  # set to -1 for the shortest possible path
            "max_chain_length": 7,
            "size": 2000,  # Generate a small-ish dataset for demonstration
        },
    }

    # Generate a friendly unique identifier and compose the file path
    unique_id = uuid.uuid4().hex[:8]
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(exist_ok=True)  # Create output directory if it doesn't exist
    jsonl_path = output_dir / f"word_ladders_{unique_id}.jsonl"

    # Step 1: Create the dataset
    if not create_dataset(jsonl_path, config):
        print("Exiting due to dataset creation failure.")
        sys.exit(1)

    # Step 2: Generate reasoning

    try:
        print("\nStep 2: Submitting reasoning batches for the dataset...")
        generate_reasoning.submit_reasoning_batches(input_path=str(jsonl_path))
    except Exception as e:
        print(f"\nError: Failed to submit reasoning batches: {str(e)}")
        sys.exit(1)

    # Step 3: Check Anthropic batch results
    # Step 4: Upload to HuggingFace 🤗

    print("\nComplete!")


if __name__ == "__main__":
    main()
