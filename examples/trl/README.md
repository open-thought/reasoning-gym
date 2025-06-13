# Training with TRL

Training stack:
- TRL for reinforcement learning training
- Accelerate (with DeepSpeed) for distributed training
- vLLM for rollouts 


## Setup

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. (Optional) Log in to Weights & Biases for experiment tracking:
```bash
# First, set your WANDB_API_KEY as an environment variable
# export WANDB_API_KEY=your_wandb_api_key

# Set the project name
# export WANDB_PROJECT=your_wandb_project_name

# Then, log in to Weights & Biases
wandb login $WANDB_API_KEY
```

3. Run the training script
```bash
# First, give execute permissions to the script
# chmod +x ./train.sh

# Then, run the training script
./train.sh
```

