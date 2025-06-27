# Chain Sum Training with veRL

This example demonstrates how to train a language model using veRL (Volcano Engine Reinforcement Learning) with the reasoning-gym environment for chain sum problems.

Requirements:

python >= 3.10

## Installation

1. **Install veRL**: Follow the installation instructions at [veRL repository](https://github.com/volcengine/verl)

2. **Install reasoning-gym**:
   ```bash
   pip install reasoning-gym
   ```

## Training

To start training the model on chain sum problems:

```bash
python grpo_train.py --config-path config --config-name grpo_trainer
```

### Configuration

You can modify the training by editing the configuration file or overriding arguments in the shell scripts directly

```bash
# Change dataset
Here it is easiest to modify the `config/grpo_trainer.yaml` file with a custom training composite. Here is an example experiment which uses a composite of algorithmic training tasks
```yaml
reasoning_gym:
  dataset_size: 20000
  developer_prompt: DeepSeekZero
  datasets:
    ab:
      weight: 1
    base_conversion:
      weight: 1
    binary_alternation:
      weight: 1
      config:
        p_solvable: 0.9
    binary_matrix:
      weight: 1
      config:
        min_n: 2
        max_n: 6
    caesar_cipher:
      weight: 1
      config:
        max_words: 10
    cryptarithm:
      weight: 1
    isomorphic_strings:
      weight: 1
      config:
        max_string_length: 8
```

**Note**: In `config/grpo_trainer.yaml` we specify default arguments to be read from `file:///workspace/verl/verl/trainer/config`. Modify this accordingly if you have a different local folder setup.

# Change configuration Set project_name and experiment_name if logging your runs to W&B. T
This config assumes a single GPU node, but you can configure this too. The following command would be for 2 GPUs, with 1 used for vLLM rollouts:

python3 -u train_grpo.py --config-paths configs/inter_generalisation --config-name algorithmic_qwen_3b \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    trainer.n_gpus_per_node=2 \
    trainer.project_name=rg-grpo \
    trainer.experiment_name=algorithmic_qwen2.5_3b

Or similarly you could define this in a config file directly
