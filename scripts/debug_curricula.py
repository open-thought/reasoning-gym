#!/usr/bin/env -S PYTHONHASHSEED=1 python3
"""Generate a markdown document showing curriculum progression for all datasets"""

import argparse
import textwrap
from copy import deepcopy
from pathlib import Path

from tqdm import tqdm

from reasoning_gym.factory import DATASETS, CURRICULA, create_curriculum, create_dataset


def generate_curricula_doc(num_examples: int = 1, show_config: bool = False) -> str:
    """Generate markdown content showing curriculum progression
    
    Args:
        num_examples: Number of examples to generate per difficulty level
        show_config: Whether to show the effective dataset configuration
    """

    # Start with header
    content = ["# Reasoning Gym Curriculum Progression\n"]
    content.append(
        "This document shows how tasks change as curriculum difficulty increases for each dataset.\n\n"
    )

    # Get datasets with curricula
    datasets_with_curricula = sorted([name for name in DATASETS.keys() if name in CURRICULA])

    # Add index
    content.append("## Available Curricula\n")
    for name in datasets_with_curricula:
        # Create anchor link
        anchor = name.replace(" ", "-").lower()
        content.append(f"- [{name}](#{anchor})\n")
    content.append("\n")

    # Add examples for each dataset with curriculum
    content.append("## Curriculum Progression Examples\n")
    
    # Create progress bar for datasets
    for name in tqdm(datasets_with_curricula, desc="Processing datasets"):
        # Add dataset header with anchor
        content.append(f"### {name}\n")

        # Get curriculum and dataset class
        curriculum = create_curriculum(name)
        dataset_cls, config_cls = DATASETS[name]

        # Get dataset class docstring if available
        try:
            dataset = create_dataset(name, seed=42)
            if dataset.__class__.__doc__:
                doc = textwrap.dedent(dataset.__class__.__doc__.strip())
                content.append(f"{doc}\n\n")
        except Exception as e:
            content.append(f"*Error loading dataset: {str(e)}*\n\n")
            continue

        # Show curriculum attributes
        content.append("#### Curriculum Attributes\n")
        for attr_name, attr in curriculum.attributes.items():
            content.append(f"- **{attr_name}**: {attr.description}\n")
            content.append(f"  - Levels: {attr.levels}\n")
        content.append("\n")

        # Show progression with all attributes increasing simultaneously
        content.append(f"#### Overall Difficulty Progression\n")
        
        # Find the maximum number of levels across all attributes
        max_levels = max(len(attr.levels) for attr in curriculum.attributes.values())
        
        # Show examples at each difficulty level
        for level in tqdm(range(max_levels), 
                         desc=f"Dataset: {name}, Overall Difficulty",
                         leave=False):
            try:
                # Reset curriculum to defaults
                curriculum = create_curriculum(name)
                
                # Set all attributes to this level, if available
                for attr_name, attr in curriculum.attributes.items():
                    if level < len(attr.levels):
                        curriculum.set_attr_level(attr_name, level)
                
                # Generate config with this level
                config = curriculum.generate_configuration({"seed": 42})
                
                # Create dataset with this config
                dataset = dataset_cls(config=config)
                
                # Show level and example
                content.append(f"##### Difficulty Level {level}\n")
                
                # Show the current values for each attribute
                content.append("Current attribute values:\n")
                for attr_name, attr in curriculum.attributes.items():
                    attr_level = min(level, len(attr.levels) - 1)
                    content.append(f"- {attr_name}: {attr.levels[attr_level]}\n")
                
                # Show the effective configuration if requested
                if show_config:
                    content.append("\nEffective configuration:\n")
                    for key, value in vars(config).items():
                        if key != "seed" and key != "size":
                            content.append(f"- {key}: {value}\n")
                
                # Generate multiple examples
                for ex_idx in range(num_examples):
                    # Get example
                    example = dataset[ex_idx]
                    
                    content.append(f"\n```\n")
                    if num_examples > 1:
                        content.append(f"Example {ex_idx + 1}:\n")
                    content.append(f"Question: {example['question']}\n")
                    content.append(f"Answer: {example['answer']}\n")
                    if example.get("metadata"):
                        content.append(f"Metadata: {example['metadata']}\n")
                    content.append("```\n")
            except Exception as e:
                content.append(f"##### Difficulty Level {level}\n")
                content.append(f"*Error generating example: {str(e)}*\n\n")
        
        content.append("\n")

    return "".join(content)


def main():
    """Generate curricula markdown file"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate curriculum documentation")
    parser.add_argument("--examples", type=int, default=1, 
                        help="Number of examples to generate per difficulty level")
    parser.add_argument("--show-config", action="store_true", 
                        help="Show the effective dataset configuration")
    parser.add_argument("--output", type=str, default="CURRICULA.md",
                        help="Output file path (relative to project root)")
    args = parser.parse_args()
    
    # Ensure scripts directory exists
    script_dir = Path(__file__).parent
    if not script_dir.exists():
        script_dir.mkdir(parents=True)

    curricula_path = script_dir.parent / args.output
    
    print(f"Generating curricula documentation...")
    print(f"Number of examples per level: {args.examples}")
    print(f"Show configuration: {args.show_config}")
    
    curricula_content = generate_curricula_doc(
        num_examples=args.examples,
        show_config=args.show_config
    )

    with open(curricula_path, "w") as f:
        f.write(curricula_content)
        f.write("\n")

    print(f"Generated curricula documentation at {curricula_path}")


if __name__ == "__main__":
    main()
