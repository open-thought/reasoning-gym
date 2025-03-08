#!/usr/bin/env -S PYTHONHASHSEED=1 python3
"""Generate a markdown document showing curriculum progression for all datasets"""

import textwrap
from copy import deepcopy
from pathlib import Path

from reasoning_gym.factory import DATASETS, CURRICULA, create_curriculum, create_dataset


def generate_curricula_doc() -> str:
    """Generate markdown content showing curriculum progression"""

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
    for name in datasets_with_curricula:
        # Add dataset header with anchor
        content.append(f"### {name}\n")

        # Get curriculum and dataset class
        curriculum = create_curriculum(name)
        dataset_cls, config_cls = DATASETS[name]

        # Get dataset class docstring if available
        dataset = create_dataset(name, seed=42)
        if dataset.__class__.__doc__:
            doc = textwrap.dedent(dataset.__class__.__doc__.strip())
            content.append(f"{doc}\n\n")

        # Show curriculum attributes
        content.append("#### Curriculum Attributes\n")
        for attr_name, attr in curriculum.attributes.items():
            content.append(f"- **{attr_name}**: {attr.description}\n")
            content.append(f"  - Levels: {attr.levels}\n")
        content.append("\n")

        # For each attribute, show progression through levels
        for attr_name, attr in curriculum.attributes.items():
            content.append(f"#### Progression of '{attr_name}'\n")
            
            # Reset curriculum to defaults
            curriculum = create_curriculum(name)
            
            # Show examples at each level
            for level in range(len(attr.levels)):
                # Set the attribute to this level
                curriculum.set_attr_level(attr_name, level)
                
                # Generate config with this level
                config = curriculum.generate_configuration({"seed": 42})
                
                # Create dataset with this config
                dataset = dataset_cls(config=config)
                
                # Get first example
                example = dataset[0]
                
                # Show level and example
                content.append(f"##### Level {level}: {attr.levels[level]}\n")
                content.append("```\n")
                content.append(f"Question: {example['question']}\n")
                content.append(f"Answer: {example['answer']}\n")
                if example.get("metadata"):
                    content.append(f"Metadata: {example['metadata']}\n")
                content.append("```\n\n")
            
            content.append("\n")

    return "".join(content)


def main():
    """Generate curricula markdown file"""
    # Ensure scripts directory exists
    script_dir = Path(__file__).parent
    if not script_dir.exists():
        script_dir.mkdir(parents=True)

    curricula_path = script_dir.parent / "CURRICULA.md"
    curricula_content = generate_curricula_doc()

    with open(curricula_path, "w") as f:
        f.write(curricula_content)
        f.write("\n")

    print(f"Generated curricula documentation at {curricula_path}")


if __name__ == "__main__":
    main()
