### env setup

```
conda create --name verl python=3.12 -y
conda activate verl

pip install flash-attn --no-build-isolation
pip install vllm==0.7.0 ray wandb
```

### clone and install veRL

tested with verl HEAD a65c9157bc0b85b64cd753de19f94e80a11bd871

```
git clone https://github.com/volcengine/verl.git
cd verl
pip install -e .
```
