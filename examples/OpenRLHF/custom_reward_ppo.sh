#!/bin/bash
args=(
   custom_reward.py
   --pretrain meta-llama/Llama-3.2-3B-Instruct  # meta-llama/Llama-3.2-1B-Instruct
   --save_path ./checkpoint/Llama-3.2-3B-rl     # ./checkpoint/Llama-3.2-1b-lr
   --save_steps -1
   --logging_steps 1
   --eval_steps -1
   --micro_train_batch_size 2
   --train_batch_size 128
   --micro_rollout_batch_size 16
   --rollout_batch_size 1024
   --max_epochs 1
   --prompt_max_len 1024
   --generate_max_len 1024
   --zero_stage 2
   --bf16
   --actor_learning_rate 5e-7
   --init_kl_coef 0.01
   --prompt_data chain_sum # leg_counting
   --input_key question
   --apply_chat_template
   --normalize_reward
   --adam_offload
   --flash_attn
   --gradient_checkpointing
   --max_samples 100000
   --critic_learning_rate 9e-6
)
# Add wandb argument only if wandb_token is set
if [[ -n "${wandb_token}" ]]; then
    args+=(--use_wandb "${wandb_token}")
fi
deepspeed ${args[@]}
