## Setup

Prepare virtual environment, e.g.

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

Login to W&B and HuggingFace if desired

```bash
wandb login
huggingface-cli login
```

## Training

Here we assume two GPUs, with one used for inference (vLLM) and the other for training (accelerate). You may need to adjust some settings for different GPU configs.

Run the vLLM server for inference:

```bash
CUDA_VISIBLE_DEVICES=0 vf-vllm --model Qwen/Qwen2.5-1.5B-Instruct --tensor-parallel-size 1
```

Run the training script using accelerate:

```bash
CUDA_VISIBLE_DEVICES=1 accelerate launch --config-file zero3.yaml --num-processes 1 vf_rg.py
```
