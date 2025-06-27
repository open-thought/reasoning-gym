# Chain Sum LORA Training with unsloth

This example demonstrates how to fine-tune an LLM with RL on a reasoning gym environment using the **unsloth** framework. Unsloth is a efficient open-source library for fine-tuning & RL. Unsloths default training path uses quantised low rank adaption (QLORA) which results in a signficantly lower memory footprint ($\approx 3x$) and means you can significantly increase batch sizes and context length without risking OOM errors.

Requirements:

python >= 3.10

## Installation

1. **Install reasoning-gym**:
   ```bash
   pip install reasoning-gym
   ```
2. **Install unsloth dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run training script**
   To start training with unsloth with RG environments using default arguments run the following:

   ```bash
   python train_grpo_lora.py
   ```

   To customise/override any default arguments you can simply:
   ```bash
   python train_grpo_lora.py  --dataset-name chain_sum --max-seq-length 512 --model-id Qwen/Qwen2.5-7B-Instruct

**Note** the free open-source version of  unsloth is currently built to train models in single GPU environments only.
