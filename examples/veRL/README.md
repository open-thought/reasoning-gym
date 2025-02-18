### env setup

```
conda create --name verl python=3.11 -y
conda activate verl

pip install flash-attn --no-build-isolation
pip install ray wandb
# pip3 install vllm==0.7.0
pip3 install vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
```

Regarding vllm>0.7 see: [docs](https://verl.readthedocs.io/en/latest/README_vllm0.7.html)


### clone and install veRL

tested with verl HEAD 0dfcb7f99e299940e1792a386df13c7591df351a

```
git clone https://github.com/volcengine/verl.git
cd verl
pip install -e .
```


Optionally log in to huggingface hub and wandb with your keys:

```
huggingface-cli login
wandb login
```
