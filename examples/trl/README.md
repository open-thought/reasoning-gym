# Training with TRL

Training stack:
- TRL for reinforcement learning training
- Accelerate (with DeepSpeed) for distributed training
- vLLM for rollouts


## Setup

This tutorial uses CUDA 11.8, Python 3.10, and PyTorch 2.5.1

Moreover, we assume that you have 2 GPUs on your machine, the last of which is used for vLLM rollouts.

If you have more than 2 GPUs, adjust the `./config/grpo.yaml` file so that the `vllm_device` is set to the last index of your GPU. For example, if you have 4 GPUs, set it to 3:
```yaml
vllm_device: 3  # If you have 4 GPUs, set this to 3
```

Moreover, you would need to update the `CUDA_VISIBLE_DEVICES` environment variable in the `train.sh` script to include all your available GPUs. For example, if you have 4 GPUs, set it to:
```bash
# ./train.sh

# ... beginning of the script
export CUDA_VISIBLE_DEVICES=0,1,2,3
# ... rest of the script
```



1. Install the required packages:
```bash
# First, give execute permissions to the script
# chmod +x ./set_env.sh

# Then, run the setup script
./set_env.sh
```

2. (Optional) Log in to Weights & Biases for experiment tracking:
```bash
# First, set your WANDB_API_KEY as an environment variable
export WANDB_API_KEY=your_wandb_api_key

# Set the project name
export WANDB_PROJECT=your_wandb_project_name
```

3. Run the training script
```bash
# First, give execute permissions to the script
# chmod +x ./train.sh

# Then, run the training script
./train.sh
```
