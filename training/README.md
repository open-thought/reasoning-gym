# Reasoning Gym Model Training

Training codebase for training LLMs using Reasoning Gym procedural dataset generators.

### Requirements

1. Prepare and activate a Python 3.11 virtual environment however you prefer.
2. Install Reasoning Gym:

```bash
cd reasoning-gym/
pip install -e .
```

3. Install training-specific Python package dependencies:

```bash
pip install flash-attn --no-build-isolation
pip install ray wandb
pip install vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
```

4. Install veRL (tested with HEAD c34206925e2a50fd452e474db857b4d488f8602d):

```bash
git clone https://github.com/volcengine/verl.git
cd verl
pip install -e .
```

5. Log in to HF and W&B:

```bash
huggingface-cli login
wandb login
```

### Usage

First, activate the virtual environment you prepared.

Example GRPO training usage:

```bash
python3 -u train_grpo.py --config-name llama3.1_1b_grpo \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    trainer.project_name=rg-test \
    trainer.experiment_name=verl_grpo_llama3.1_1b \
    trainer.n_gpus_per_node=2 $@ 2>&1 | tee verl_output.log
```

Then, having saved this as a bash script such as `train.sh`, run it:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash train.sh
```

CUDA_VISIBLE_DEVICES is set to 0,1 to use the first two GPUs on the machine (see `nvidia-smi` output). This can be adjusted as needed. `tensor_model_parallel_size` and `n_gpus_per_node` should also be set to the number of GPUs you are using.

You can change all configuration options by either modifying the config YAML (in this case, `config/llama3.1_1b_grpo.yaml`) or providing them as arguments to the Python script. Note that the batch sizes set in the Llama 1B and Qwen 1.5B configs are as high as it was possible for me to set them for the puzzles dataset mix on 2xA6000 GPUs without OOMs. Depending on the hardware you use and the datasets you train on, you may need to adjust these.
