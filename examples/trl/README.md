# TRL Examples

This directory contains examples using the [TRL (Transformer Reinforcement Learning) library](https://github.com/huggingface/trl) to fine-tune language models with reinforcement learning techniques.

## GRPO Example

The main example demonstrates using GRPO (Group Relative Policy Optimization) to fine-tune a language model on reasoning tasks from reasoning-gym. It includes:

- Custom reward functions for answer accuracy and format compliance
- Integration with reasoning-gym datasets
- Configurable training parameters via YAML config
- Wandb logging and model checkpointing
- Evaluation on held-out test sets

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Configure the training parameters in `config/grpo.yaml`
2. Run the training script:

```bash
python main_grpo_reward.py
```

The model will be trained using GRPO with the specified reasoning-gym dataset and evaluation metrics will be logged to Weights & Biases.
