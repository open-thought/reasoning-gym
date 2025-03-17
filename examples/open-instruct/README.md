# Reasoning Gym Open-Instruct Example

This example demonstrates how to use [open-instruct](https://github.com/allenai/open-instruct) with **reasoning-gym** for training language models through reinforcement learning.

## Environment Setup

Before running the training script, you may want to set the following environment variables:

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

- `CUDA_VISIBLE_DEVICES=0` specifies which GPUs to use (in this case, GPUs 0 )
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` enables dynamic GPU memory allocation

## Running the Training Script

To start training, run:

```bash
bash grpo_config.sh
```

This will train a DeepSeek-R1-Distill-Qwen-1.5B model using GRPO (Generative Reinforcement Policy Optimization) on reasoning tasks.

## Key Features

- Uses vLLM for efficient inference
- Supports multi-GPU training
- Implements reward functions for both answer correctness and response formatting
- Includes evaluation during training
- Supports model checkpointing and pushing to Hugging Face Hub
- Integrates with Weights & Biases for experiment tracking

