# Qwen2.5-3B-Instruct
export CUDA_VISIBLE_DEVICES="0"
PROMPT_TYPE="qwen25-math-cot"
MODEL_NAME_OR_PATH="Qwen/Qwen2.5-3B-Instruct"
bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH

# Qwen2.5-3B-Instruct-rl-finetuned
export CUDA_VISIBLE_DEVICES="0"
PROMPT_TYPE="qwen25-rg-cot"
MODEL_NAME_OR_PATH="/home/ubuntu/projects/math-eval/Qwen/Qwen-2.5-3B-rl-600"
bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH