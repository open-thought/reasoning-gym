#!/bin/bash


MAMBA_ENV="tina_eval"
eval "$(mamba shell hook --shell bash)" && mamba activate "${MAMBA_ENV}"
echo "START TIME: $(date)"
echo "PYTHON ENV: $(which python)"

source "./scripts/set/set_vars.sh"

export CUDA_VISIBLE_DEVICES="0" # NOTE: update this if you have more than 1 GPU
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
echo ""
echo "GPU_COUNT: $GPU_COUNT"
echo ""

# MODEL_LIST=("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" "starzmustdie/DeepSeek-R1-Distill-Qwen-1.5B-rg-math" "agentica-org/DeepScaleR-1.5B-Preview" "knoveleng/Open-RS3" "RUC-AIBOX/STILL-3-1.5B-preview")
MODEL_LIST=("Qwen/Qwen2.5-3B-Instruct" "starzmustdie/Qwen2.5-3B-Instruct")
TASKS=("aime24" "aime25" "amc23" "minerva" "math_500")

for MODEL_NAME in "${MODEL_LIST[@]}"; do
    for TASK in "${TASKS[@]}"; do
        MODEL_ARGS="model_name=$MODEL_NAME,dtype=bfloat16,data_parallel_size=$GPU_COUNT,max_model_length=32768,gpu_memory_utilization=0.7,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
        lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
            --custom-tasks ./scripts/training/run_post_train_eval.py \
            --use-chat-template \
            --output-dir "${OUTPUT_DIR}/${TASK}"
    done
done

echo "END TIME: $(date)"
echo "DONE"
