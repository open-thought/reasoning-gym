"""
create_word_ladders.py – Generates the word ladder dataset as a JSONL file.
Each line is a JSON object with keys: question, answer, metadata, and reasoning (set to None).
"""

import json
import uuid
from pathlib import Path

from tqdm import tqdm

import reasoning_gym


def check_duplicates(jsonl_path: str) -> tuple[bool, dict]:
    """
    Check for duplicate word pairs in a word ladder JSONL file.

    Returns:
        tuple[bool, dict]: (has_duplicates, valid_entries) where:
            - has_duplicates: True if any duplicates were found
            - valid_entries: Dict mapping line_number -> data for non-duplicate entries

    Note: A pair is considered duplicate if either (word1, word2) or (word2, word1)
    already exists, since word ladder paths are bidirectional.
    """
    pairs_seen = {}  # (start, end) -> (line_number, data)
    valid_entries = {}
    duplicates_found = False

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            data = json.loads(line)
            metadata = data["metadata"]
            pair = (metadata["start_word"], metadata["end_word"])
            reverse_pair = (metadata["end_word"], metadata["start_word"])

            # Check both orientations of the pair
            if pair in pairs_seen or reverse_pair in pairs_seen:
                duplicates_found = True
                # Skip this entry - it's a duplicate
                continue
            else:
                # Store both the line number and data for valid entries
                pairs_seen[pair] = (line_num, data)
                valid_entries[line_num] = data

    return duplicates_found, valid_entries


def create_word_ladder_dataset(jsonl_path: str = None, config: dict = None) -> None:
    """
    Creates a word ladder dataset and writes each sample as a JSON line.
    Ensures no duplicate word pairs by regenerating as needed.
    """
    if config is None:
        raise ValueError("Configuration (config) must be provided.")

    # Create output directory if it doesn't exist.
    # Updated path to point to the parent folder's output.
    output_dir = Path(__file__).resolve().parent.parent / "output"
    output_dir.mkdir(exist_ok=True)

    # Determine the file path based on uuid when not provided
    if jsonl_path is None:
        unique_id = uuid.uuid4().hex[:8]
        jsonl_path = output_dir / f"word_ladders_{unique_id}.jsonl"
    else:
        jsonl_path = Path(jsonl_path)

    target_size = config["dataset_config"]["size"]
    current_size = 0
    max_attempts = 3  # Limit total regeneration attempts
    attempt = 0

    # Initial generation
    dataset = reasoning_gym.create_dataset(config["dataset_name"], **config["dataset_config"])
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in tqdm(dataset, desc="Generating initial ladder examples"):
            row = {
                "question": item["question"],
                "answer": item["answer"],
                "reasoning": None,
                "metadata": item.get("metadata", {}),
            }
            f.write(json.dumps(row) + "\n")

    while attempt < max_attempts:
        # Check entire file for duplicates
        has_duplicates, valid_entries = check_duplicates(jsonl_path)
        current_size = len(valid_entries)

        if not has_duplicates and current_size == target_size:
            print(f"\nSuccessfully created dataset with {current_size} unique examples.")
            return

        # If we have duplicates or not enough entries, regenerate the missing amount
        needed = target_size - current_size
        if needed > 0:
            print(f"\nAttempt {attempt + 1}: Regenerating {needed} examples to replace duplicates/missing entries...")

            # Generate additional examples
            config["dataset_config"]["size"] = needed
            additional_dataset = reasoning_gym.create_dataset(config["dataset_name"], **config["dataset_config"])

            # Write all entries to a temporary file
            temp_path = jsonl_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                # Write existing valid entries
                for data in valid_entries.values():
                    f.write(json.dumps(data) + "\n")

                # Write new entries
                for item in additional_dataset:
                    row = {
                        "question": item["question"],
                        "answer": item["answer"],
                        "reasoning": None,
                        "metadata": item.get("metadata", {}),
                    }
                    f.write(json.dumps(row) + "\n")

            # Replace original file with temporary file
            temp_path.replace(jsonl_path)

            # Note: We'll check for duplicates again at the start of the next loop

        attempt += 1

    if current_size < target_size:
        print(f"\nWarning: Could only generate {current_size} unique examples after {max_attempts} attempts.")
    else:
        print(f"\nSuccessfully created dataset with {current_size} unique examples.")
