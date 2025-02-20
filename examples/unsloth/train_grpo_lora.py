"""
Minimal example using Unsloth and vLLM for efficient GRPO training of a model with (Q)LoRA.

Adapted from Unsloth's documentation examples.
"""

from unsloth import FastLanguageModel, PatchFastRL

PatchFastRL("GRPO", FastLanguageModel)

import argparse
import logging
import re

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from trl import GRPOConfig, GRPOTrainer
from unsloth import is_bfloat16_supported

import reasoning_gym
from reasoning_gym import utils


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


def get_model_and_tokenizer(model_id, max_seq_length, lora_rank, quantize, gpu_memory_utilization) -> tuple:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=max_seq_length,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=gpu_memory_utilization,
        load_in_4bit=quantize,
        fast_inference=True,
    )

    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    model = FastLanguageModel.get_peft_model(
        model, r=lora_rank, target_modules=target_modules, lora_alpha=lora_rank, use_gradient_checkpointing="unsloth"
    )

    return model, tokenizer


class GRPOTrainerCustom(GRPOTrainer):
    def __init__(self, model, args: GRPOConfig, tokenizer, train_dataset: Dataset):
        super().__init__(
            model,
            reward_funcs=[self._accuracy_reward, self._format_reward],
            args=args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
        )

    def _format_reward(self, completions, **kwargs):
        regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"
        matches = [re.match(regex, completion, flags=re.DOTALL) for completion in completions]
        return [1.0 if match else 0.0 for match in matches]

    def _accuracy_reward(self, completions, metadata, **kwargs):
        answers = [utils.extract_answer(completion) for completion in completions]
        return [self.train_dataset.data.score_answer(answer, entry=obj) for (answer, obj) in zip(answers, metadata)]


def train(model, tokenizer, dataset, training_args):
    trainer = GRPOTrainerCustom(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()

    logging.info("Saving model...")
    trainer.save_model("outputs")


def evaluate(model, tokenizer, dataset, *args, **kwargs):
    model.eval()
    correct_preds = 0
    total_preds = 0

    for i in tqdm(range(len(dataset))):
        item = dataset[i]
        prompt = item["prompt"]
        metadata = item["metadata"]
        inputs = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                pad_token_id=tokenizer.eos_token_id,
                *args,
                **kwargs,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = utils.extract_answer(generated_text)
        score = dataset.data.score_answer(answer, entry=metadata)
        correct_preds += score
        total_preds += 1

    return correct_preds / total_preds


def main(args):
    model, tokenizer = get_model_and_tokenizer(
        args.model_id, args.max_seq_length, args.lora_rank, args.quantize, args.gpu_memory_utilization
    )

    developer_prompt = utils.SYSTEM_PROMPTS["DeepSeekZero"]
    dataset = ReasoningGymDataset(args.dataset_name, args.dataset_seed, args.dataset_size, tokenizer, developer_prompt)

    training_args = GRPOConfig(
        output_dir="outputs",
        use_vllm=True,
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=1,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=1,
        num_generations=args.num_generations,
        num_train_epochs=args.train_epochs,
        save_steps=100,
        max_grad_norm=0.1,
    )

    train(model, tokenizer, dataset, training_args)

    model = FastLanguageModel.for_inference(model)

    eval_dataset = ReasoningGymDataset(
        args.dataset_name,
        args.eval_seed,
        args.eval_size,
        tokenizer,
        utils.SYSTEM_PROMPTS["DeepSeekZero"],
    )

    accuracy = evaluate(model, tokenizer, eval_dataset, max_new_tokens=training_args.max_completion_length)
    logging.info(f"Evaluation accuracy: {accuracy * 100}%")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--dataset-name", type=str)

    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--train-epochs", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=8)

    parser.add_argument("--dataset-seed", type=int, default=42)
    parser.add_argument("--dataset-size", type=int, default=1000)

    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--eval-size", type=int, default=100)

    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)

    args = parser.parse_args()
    main(args)
