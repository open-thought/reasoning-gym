#!/bin/bash


MAMBA_ENV="tina_eval"
eval "$(mamba shell hook --shell bash)" && mamba activate "${MAMBA_ENV}"
echo "START TIME: $(date)"
echo "PYTHON ENV: $(which python)"

source "./scripts/set/set_vars.sh"

export CUDA_VISIBLE_DEVICES=0,1 # make sure all evaluation run on 2 GPUs
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")

echo ""
echo "GPU_COUNT: $GPU_COUNT, make sure using 2 GPUs."
echo ""

MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
PT_TYPE="grpo"

## Main datasets
DATASET_NAME="curated_deepscaler"
#DATASET_NAME="curated_still"
#DATASET_NAME="curated_open_rs3"
#DATASET_NAME="curated_open_rs2"
#DATASET_NAME="curated_open_rs1"

## Extra datasets
#DATASET_NAME="curated_limr"
#DATASET_NAME="curated_open_r1"
#DATASET_NAME="curated_thoughts"

## Ablation
#DATASET_NAME="curated_limr_large_lr_ablation"
#DATASET_NAME="curated_limr_small_lr_ablation"
#DATASET_NAME="curated_limr_large_rank_ablation"
#DATASET_NAME="curated_limr_medium_rank_ablation"
#DATASET_NAME="curated_limr_small_rank_ablation"
#DATASET_NAME="curated_limr_tiny_rank_ablation"
#DATASET_NAME="curated_open_rs3_drgrpo_ablation"

CKPT_LIST=$(ls "${CKPT_DIR}/models/${MODEL_NAME}/${PT_TYPE}_${DATASET_NAME}" | grep -E "^checkpoint-[0-9]+$")
#CKPT_LIST=("checkpoint-XXX")

# loop over all the checkpoints in the list
for CKPT in "${CKPT_LIST[@]}"; do
    echo "Running model post train merging base and adapter for checkpoint: ${CKPT}"
    python ./scripts/training/run_post_train_merge.py \
      --model_name "${MODEL_NAME}" \
      --adapter_type "${PT_TYPE}_${DATASET_NAME}" \
      --ckpt "${CKPT}"

    MODEL_PATH="${CKPT_DIR}/models/${MODEL_NAME}/${PT_TYPE}_${DATASET_NAME}/${CKPT}-merged"

    # Set model arguments (ensure that MODEL_PATH, GPU_COUNT, OUTPUT_DIR, and MODEL are defined)
    MODEL_ARGS="pretrained=$MODEL_PATH,dtype=bfloat16,data_parallel_size=$GPU_COUNT,max_model_length=32768,gpu_memory_utilization=0.5,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"

    # Define an array of tasks to evaluate
    tasks=("aime24" "math_500" "gpqa:diamond" "aime25" "amc23" "minerva")

    # Loop over each task and evaluate
    for TASK in "${tasks[@]}"; do
      echo "Evaluating task: $TASK"
      lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
          --custom-tasks ./scripts/training/run_post_train_eval.py \
          --use-chat-template \
          --output-dir "${OUTPUT_DIR}/${MODEL}/${TASK}"
    done

done

echo "END TIME: $(date)"
echo "DONE"
