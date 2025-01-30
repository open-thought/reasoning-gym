#!/usr/bin/env python3
"""Generate a markdown gallery of all available datasets with examples"""

import os
import textwrap
from pathlib import Path

import reasoning_gym.cognition.figlet_fonts
import reasoning_gym.cognition.rubiks_cube
from reasoning_gym.factory import DATASETS, create_dataset


def generate_gallery() -> str:
    """Generate markdown content for the gallery"""

    # Start with header
    content = ["# Reasoning Gym Dataset Gallery\n"]
    content.append("This gallery shows examples from all available datasets using their default configurations.\n\n")

    # Add index
    content.append("## Available Datasets\n")
    for name in sorted(DATASETS.keys()):
        # Create anchor link
        anchor = name.replace("_", "-").lower()
        content.append(f"- [{name}](#{anchor})\n")
    content.append("\n")

    # Add examples for each dataset
    content.append("## Dataset Examples\n")
    for name in sorted(DATASETS.keys()):
        dataset = create_dataset(name)

        # Add dataset header with anchor
        anchor = name.replace("_", "-").lower()
        content.append(f"### {name}\n")

        # Get dataset class docstring if available
        if dataset.__class__.__doc__:
            doc = textwrap.dedent(dataset.__class__.__doc__.strip())
            content.append(f"{doc}\n\n")

        # Show configuration
        content.append("Default configuration:\n")
        content.append("```python\n")
        for key, value in dataset.config.__dict__.items():
            if not key.startswith("_"):
                content.append(f"{key} = {value}\n")
        content.append("```\n\n")

        # Show examples
        content.append("Example tasks:\n")
        content.append("```\n")
        for i, item in enumerate(dataset):
            if i >= 3:
                break
            content.append(f"Example {i+1}:\n")
            content.append(f"Question: {item['question']}\n")
            content.append(f"Answer: {item['answer']}\n")
            if item.get("metadata"):
                content.append(f"Metadata: {item['metadata']}\n")
            content.append("\n")
        content.append("```\n\n")

    return "".join(content)


def main():
    """Generate gallery markdown file"""
    # Ensure scripts directory exists
    script_dir = Path(__file__).parent
    if not script_dir.exists():
        script_dir.mkdir(parents=True)

    gallery_path = script_dir.parent / "GALLERY.md"
    gallery_content = generate_gallery()

    with open(gallery_path, "w") as f:
        f.write(gallery_content)

    print(f"Generated gallery at {gallery_path}")


if __name__ == "__main__":
    main()
