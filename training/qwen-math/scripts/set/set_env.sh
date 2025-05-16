#!/bin/bash
# python 3.10 + cuda 11.8.0
# the execution order the following commands matter

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

conda clean -a -y
mamba clean -a -y
pip install --upgrade pip
pip cache purge

# cuda, gcc/g++, torch
# mamba install cuda -c nvidia/label/cuda-11.8.0 -y
# mamba install gcc gxx -c conda-forge -y
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118

# xformers
pip install xformers==0.0.28.post3 --index-url https://download.pytorch.org/whl/cu118

# vLLM pre-compiled with CUDA 11.8
pip install https://github.com/vllm-project/vllm/releases/download/v0.7.2/vllm-0.7.2+cu118-cp38-abi3-manylinux1_x86_64.whl

pip install deepspeed
pip install flash-attn==2.7.3 --no-build-isolation
pip install peft

pip install "trl==0.15.2"
pip install latex2sympy2_extended
pip install "math_verify==0.5.2"
pip install word2number
pip install scipy

pip install wandb
pip install plotly
pip install matplotlib
pip install seaborn
