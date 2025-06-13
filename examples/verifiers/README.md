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

Run the training script

```bash
python train.py
```
