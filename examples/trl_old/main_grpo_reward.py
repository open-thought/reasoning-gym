# This example is an adapted version of HuggingFace trl GRPO code:
# link : https://github.com/huggingface/open-r1/blob/main/src/open_r1/grpo.py
import logging
import os
import re
import sys
from dataclasses import dataclass
from typing import Optional

import datasets
import torch
import transformers
from grpo_config import ScriptArguments
from peft import LoraConfig
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser

import reasoning_gym
from reasoning_gym.utils import extract_answer


class ReasoningGymDataset(Dataset):
    def __init__(self, dataset_name, seed, size, tokenizer, developer_prompt, developer_role="system") -> None:
        super().__init__()
        self.data = reasoning_gym.create_dataset(dataset_name, seed=seed, size=size)
        self.tokenizer = tokenizer
        self.developer_role = developer_role
        self.developer_prompt = developer_prompt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["question"]

        chat = []

        if self.developer_role is not None:
            chat.append({"role": self.developer_role, "content": self.developer_prompt})
        chat.append({"role": "user", "content": question})

        prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        return {"prompt": prompt, "metadata": item}


class GRPOTrainerCustom(GRPOTrainer):
    def __init__(
        self,
        model,
        dataset_name,
        args: GRPOConfig,
        tokenizer,
        peft_config,
        seed,
        size,
        developer_role="system",
    ):
        super().__init__(
            model,
            reward_funcs=[self._accuracy_reward, self._format_reward],
            args=args,
            processing_class=tokenizer,
            peft_config=peft_config,
        )
        developer_prompt = reasoning_gym.utils.SYSTEM_PROMPTS["DeepSeekZero"]
        self.train_dataset = ReasoningGymDataset(dataset_name, seed, size, tokenizer, developer_prompt, developer_role)

    def _format_reward(self, completions, **kwargs):
        regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"
        matches = [re.match(regex, completion, flags=re.DOTALL) for completion in completions]
        return [1.0 if match else 0.0 for match in matches]

    def _accuracy_reward(self, completions, metadata, **kwargs):
        answers = [extract_answer(completion) for completion in completions]
        return [self.train_dataset.data.score_answer(answer, entry=obj) for (answer, obj) in zip(answers, metadata)]


def main(script_args, training_args, model_args):
    set_seed(training_args.seed)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger = logging.getLogger(__name__)

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)  # Set for module-level logger

    # Configure third-party library log levels
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Training arguments: {training_args}")
    logger.info(f"Model arguments: {model_args}")
    logger.info(f"Script arguments: {script_args}")

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        r=16,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
    )

    trainer = GRPOTrainerCustom(
        model,
        dataset_name=script_args.dataset_name,
        args=training_args,
        tokenizer=tokenizer,
        peft_config=peft_config,
        seed=training_args.seed,
        size=script_args.train_size,
    )

    # Training loop
    logger.info("Training model...")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is None:
        checkpoint = model.save_pretrained(training_args.output_dir)

    train_results = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_results.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": list(script_args.dataset_name),
        "dataset_tags": list(script_args.dataset_name),
        "tags": ["reasoning-gym"],
    }

    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    def evaluate_model(model, tokenizer, dataset, *args, **kwargs):
        model.eval()
        correct_preds = 0
        total_preds = 0

        for i in range(len(dataset)):
            item = dataset[i]
            prompt = item["prompt"]
            metadata = item["metadata"]
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=training_args.max_completion_length,
                    pad_token_id=tokenizer.eos_token_id,
                    *args,
                    **kwargs,
                )

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = reasoning_gym.utils.extract_answer(generated_text)
            score = dataset.data.score_answer(answer, entry=metadata)
            correct_preds += score
            total_preds += 1

        return correct_preds / total_preds

    ## Evaluate model
    logger.info("Evaluating model...")
    eval_dataset = ReasoningGymDataset(
        script_args.dataset_name,
        training_args.eval_seed,
        script_args.eval_size,
        tokenizer,
        reasoning_gym.utils.SYSTEM_PROMPTS["DeepSeekZero"],
    )

    eval_results = evaluate_model(model, tokenizer, eval_dataset)
    trainer.log_metrics("eval", {"accuracy": eval_results})
    trainer.save_metrics("eval", {"accuracy": eval_results})
    logger.info(f"Evaluation results: {eval_results}")

    if training_args.push_to_hub:
        logging.info("Pushing model to hub...")
        trainer.push_to_hub()


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
