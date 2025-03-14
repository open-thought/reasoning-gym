# Reasoning Gym Model Training

Training codebase for training LLMs using Reasoning Gym procedural dataset generators.

Example GRPO training usage:

```bash
python3 -u train_grpo.py config_name=llama3.1_1b_grpo \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    trainer.project_name=verl_chain_sum_grpo \
    trainer.experiment_name=chain_sum_llama \
    trainer.n_gpus_per_node=2 $@ 2>&1 | tee verl_output.log
```

For other configuration options, see `config/llama3.1_1b_grpo.yaml`.
