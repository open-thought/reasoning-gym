#!/bin/bash


#SBATCH --job-name=grpo_multinode
#SBATCH -D .
#SBATCH --partition=TODO
#SBATCH --account=TODO
#SBATCH --output=output-%x.%j
#SBATCH --error=error-%x.%j
#SBATCH --nodes=2                   # number of nodes
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --gres=gpu:2           # number of GPUs per node
#SBATCH --cpus-per-task=8          # number of cores per tasks
#SBATCH --mem=128G
#SBATCH --time=48:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --comment "Key=Monitoring,Value=ON"
#SBATCH --exclusive

######################
### Set environment ##
######################

ulimit -s unlimited

MAMBA_ENV="tina"
eval "$(mamba shell hook --shell bash)" && mamba activate "${MAMBA_ENV}"
echo "START TIME: $(date)"
echo "PYTHON ENV: $(which python)"

source "./scripts/set/set_vars.sh"
export GPUS_PER_NODE=2
######################

######################
#### Set network #####
######################
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
######################

export LAUNCHER="accelerate launch \
    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    --machine_rank $SLURM_NODEID \
    --rdzv_backend c10d \
    --main_process_ip $head_node_ip \
    --main_process_port 29500 \
    "

PY_SCRIPT="./tina/post_train_hf/grpo.py"
PY_CONFIG="./recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/model_curated_deepscaler.yaml"

# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="$LAUNCHER $PY_SCRIPT --config $PY_CONFIG"
srun $CMD
