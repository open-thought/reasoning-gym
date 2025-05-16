#!/bin/bash
# python 3.11 & cuda 11.8

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

conda clean -a -y
mamba clean -a -y
pip install --upgrade pip
pip cache purge

# mamba install cuda -c nvidia/label/cuda-11.8.0 -y
# mamba install gcc gxx -c conda-forge -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install xformers --index-url https://download.pytorch.org/whl/cu118
pip install vllm
pip install flash-attn --no-build-isolation

pip install accelerate
pip install datasets
pip install deepspeed
pip install distilabel[vllm,ray,openai]
pip install e2b-code-interpreter
pip install einops
pip install flake8
pip install huggingface_hub
pip install hf_transfer
pip install isort
pip install langdetect
pip install latex2sympy2_extended
pip install liger_kernel
pip install "math_verify==0.5.2"
pip install packaging
pip install parameterized
pip install peft
pip install pytest
pip install python-dotenv
pip install ruff
pip install safetensors
pip install sentencepiece
pip install transformers
pip install trl@git+https://github.com/huggingface/trl.git
pip install wandb
