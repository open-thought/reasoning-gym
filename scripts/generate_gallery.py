#!/usr/bin/env -S PYTHONHASHSEED=1 python3
"""Generate a markdown gallery of all available datasets with examples"""

import textwrap
from pathlib import Path

from reasoning_gym.factory import DATASETS, create_dataset, has_curriculum


def generate_gallery() -> str:
    """Generate markdown content for the gallery"""

    # Start with header
    content = ["# Reasoning Gym Dataset Gallery\n"]
    content.append("This gallery shows examples from all available datasets using their default configurations.\n\n")

    # Add index
    content.append(f"## Available Datasets ({len(DATASETS)})\n")
    content.append("Legend: ✅ = Has curriculum, ❌ = No curriculum\n\n")
    for name in sorted(DATASETS.keys()):
        # Create anchor link
        anchor = name.replace(" ", "-").lower()
        has_curr = "✅" if has_curriculum(name) else "❌"
        content.append(f"- [{name}](#{anchor}) {has_curr}\n")
    content.append("\n")

    # Add examples for each dataset
    content.append("## Dataset Examples\n")
    for name in sorted(DATASETS.keys()):
        if name == "composite":
            continue

        dataset = create_dataset(name, seed=42)

        # Add dataset header with anchor
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
        content.append("````\n")
        for i, item in enumerate(dataset):
            if i >= 3:
                break
            content.append(f"Example {i+1}:\n")
            content.append(f"Question: {item['question']}\n")
            content.append(f"Answer: {item['answer']}\n")
            if item.get("metadata"):
                content.append(f"Metadata: {item['metadata']}\n")
            content.append("\n")
        content.append("````\n\n")

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
        f.write("\n")

    print(f"Generated gallery at {gallery_path}")


if __name__ == "__main__":
    main()
