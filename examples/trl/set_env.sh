#!/bin/bash
# python 3.10 + cuda 11.8.0
# the execution order the following commands matter

conda clean -a -y
pip install --upgrade pip
pip cache purge

# torch
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118

# xformers
pip install xformers==0.0.28.post3 --index-url https://download.pytorch.org/whl/cu118

# vLLM pre-compiled with CUDA 11.8
pip install https://github.com/vllm-project/vllm/releases/download/v0.7.2/vllm-0.7.2+cu118-cp38-abi3-manylinux1_x86_64.whl

pip install deepspeed
pip install flash-attn==2.7.3 --no-build-isolation
pip install "trl==0.15.2"
pip install "transformers==4.49.0"
pip install wandb
pip install reasoning-gym
