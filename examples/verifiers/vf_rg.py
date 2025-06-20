"""Example training script for using the Reasoning Gym environment in verifiers."""

import verifiers as vf
from verifiers.envs.reasoninggym_env import ReasoningGymEnv

model_name = f"Qwen/Qwen2.5-1.5B-Instruct"
model, tokenizer = vf.get_model_and_tokenizer(model_name)

vf_env = ReasoningGymEnv(
    gym=[
        "basic_arithmetic",
        "bitwise_arithmetic",
        "decimal_arithmetic",
    ],
    num_samples=100,
    num_eval_samples=50,
    max_concurrent=100,
)

training_args = vf.grpo_defaults(run_name="reasoning-gym-test")
training_args.num_iterations = 1
training_args.per_device_train_batch_size = 4
training_args.num_generations = 8
training_args.gradient_accumulation_steps = 4
training_args.max_prompt_length = 1024
training_args.max_completion_length = 4096
training_args.max_steps = 100

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=training_args,
)

trainer.train()
