#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
NUM_PROCESSES_TRAINING=$((GPU_COUNT - 1))

echo ""
echo "Number of GPUs: ${GPU_COUNT}"
echo "Number of processes for training: ${NUM_PROCESSES_TRAINING}"
echo ""

PY_SCRIPT="./grpo.py"
PY_CONFIG="./config/grpo.yaml"
ACCELERATE_DS_CONFIG="./config/ds_zero2.yaml"

echo "START TIME: $(date)"

export WANDB_PROJECT="reasoning-gym-trl"

accelerate launch \
    --config_file "${ACCELERATE_DS_CONFIG}" \
    --main_process_port=29500 \
    --num_processes="${NUM_PROCESSES_TRAINING}" "${PY_SCRIPT}" --config "${PY_CONFIG}"

echo "END TIME: $(date)"
echo "DONE"
