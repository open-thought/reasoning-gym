# Contributing to Reasoning Gym

Thank you for your interest in contributing to Reasoning Gym! This document provides guidelines and instructions for contributing to the project.

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/open-thought/reasoning-gym.git
   ```

2. Create a virtual environment (using conda):
   ```bash
   conda create --name reasoning_gym python=3.11 -y
   conda activate reasoning_gym
   ```

3. Install the package in editable mode:
   ```bash
   pip install -e .
   ```

4. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

## Creating Procedural Datasets

When creating new datasets, please follow these guidelines:

1. **Focus on Complex Problems**: 
   - Prioritize problems where guessing has a low probability of success (e.g., number multiplication)
   - Avoid tasks with small answer sets (true/false, multiple-choice) as they create noisy rewards for RL

2. **Implementation Requirements**:
   - Create a configuration class
   - Derive your dataset class from `ProceduralDataset` (see [dataset.py](https://github.com/open-thought/reasoning-gym/blob/main/reasoning_gym/dataset.py))
   - Include comprehensive unit tests
   - Return dictionary items with keys: `"question"`, `"answer"`, and `"metadata"`
   - Use `None` for `"answer"` when multiple valid answers exist
   - For complex datasets, implement the `score_answer()` method (return value range: [0, 1])

3. **Getting Started**:
   - Review example implementations:
     - [chain_sum.py](reasoning_gym/arithmetic/chain_sum.py)
     - [test_chain_sum.py](https://github.com/open-thought/reasoning-gym/blob/main/tests/test_chain_sum.py)
   - Write clear question prompts that an average human can understand and answer correctly

## Pull Request Process

1. **Fork and Clone**:
   - [Fork the repository](https://docs.github.com/en/get-started/quickstart/fork-a-repo)
   - Clone your fork locally
   - Read more about [forks](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/about-forks)

2. **Create a Feature Branch**:
   - Work on a [new branch](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-and-deleting-branches-within-your-repository)
   - Keep changes focused and minimal

3. **Code Quality**:
   - Install pre-commit hooks: `pre-commit install`
   - Run `pre-commit run -a` before committing
   - When using AI coding assistants (cursor, aider, etc.), ensure proper formatting

4. **Submit Your PR**:
   - [Create a Pull Request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork)
   - [Request review](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/requesting-a-pull-request-review)
   - Do not include changes to `GALLERY.md` (it's updated automatically)

5. **Review Process**:
   - Address reviewer feedback promptly
   - Keep discussions constructive
   - Once approved, your changes will be merged into `main`

## Need Help?

Join our community discussion in the `#reasoning-gym` channel on the [GPU-Mode Discord server](https://discord.gg/gpumode).


