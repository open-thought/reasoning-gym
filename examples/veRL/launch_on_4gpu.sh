#!/bin/bash

export N_GPUS=4
export BASE_MODEL=meta-llama/Llama-3.2-1B-Instruct
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=chain_sum_llama
export VLLM_ATTENTION_BACKEND=XFORMERS

bash ./train.sh
