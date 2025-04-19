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
pip install ray wandb
pip install torch==2.6.0
```

4. Install veRL (tested with HEAD c34206925e2a50fd452e474db857b4d488f8602d):

```bash
pip install git+https://github.com/volcengine/verl.git@c6dc8b73cf011aa75b8c6a47b0322f50aed800ad#egg=verl
```

5. Install vLLM:

```bash
pip install vllm==0.6.3 transformers==4.50.3 fire==0.7.0
```
6. Install flash attention
```
pip install flash-attn --no-build-isolation
```

6. Log in to HF and W&B:

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


# Exporting from FSDP checkpoint to HF model checkpoint

After training your model the weights are saved across as a sharded checkpoints across several files. To faciliate simple evaluation of your trained model you may want to convert this into a HF model checkpoint. We have added a utility script to convert your sharded checkpoint into a hf checkpoint.

To run this script. Navigate to the training directory and run the following

```python
python load_fsdp_to_hf.py /path/to/fsdp/checkpoint/global_step_num/actor /path/to/hugginface/checkpoint/global_step_num/actor/huggingface saved_model_name
```

For example

```python
python utils/load_fsdp_to_hf.py checkpoints/rg-test/intra_reasoning_algorithmic_qwen_3b_composite/global_step_400/actor/ checkpoints/rg-test/intra_reasoning_algorithmic_qwen_3b_composite/global_step_400/actor/huggingface qwen3b
```

# Run evaluations

From here you may to run evaluations of your trained model. In the `training/evaluation` directory there is a script `evaluate_model.py` which you csn run to evaluate your trained model on a specific dataset. You specify evaluation parameters in a yaml file. This evaluation can point to either a local or remote model. For example the configuration file `training/evaluation/eval_algorithmic_composite.yaml` specifies the path to a local model which is stored as a hugginface checkpoint at `training/utils/qwen3b_500` (note that you have to convert to fsdp checkpoint to hf checkpoint for evaluation script to work as shown in the previous step).

## Run the script

```
export VLLM_ATTENTION_BACKEND=XFORMERS
```

Navigate to evaluations directory:
```
python evaluate_model.py --config path-to-yaml
```
For example
```
python evaluate_model.py --config eval_algorithmic_composite.yaml
```
