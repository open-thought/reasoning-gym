import logging
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser

import reasoning_gym
from reasoning_gym.coaching.experiment import Experiment
from reasoning_gym.composite import DatasetSpec
from reasoning_gym.dataset import ProceduralDataset
from reasoning_gym.utils import SYSTEM_PROMPTS, extract_answer


@dataclass
class DatasetConfigItem:
    weight: Optional[float] = field(default=1.0)
    config: Optional[dict] = field(default_factory=dict)


@dataclass
class DatasetConfig:
    dataset_size: int = field(default=1000)
    developer_prompt: str = field(default="DeepSeekZero")
    developer_role: str = field(default="system")
    datasets: dict[str, DatasetConfigItem] = field(default=None)

    def __post_init__(self):
        # Convert dictionary items to DatasetConfigItem instances
        if self.datasets:
            converted_datasets = {}
            for name, config_item in self.datasets.items():
                if isinstance(config_item, dict):
                    converted_datasets[name] = DatasetConfigItem(**config_item)
                else:
                    converted_datasets[name] = config_item
            self.datasets = converted_datasets


class ReasoningGymDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        procedural_dataset: Optional[ProceduralDataset] = None,
        experiment: Optional[Experiment] = None,
        developer_prompt: Optional[str] = None,
        developer_role: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.data = procedural_dataset or experiment.composite
        self.experiment = experiment
        self.developer_prompt = developer_prompt
        self.developer_role = developer_role

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        question = item["question"]

        chat = []
        if self.developer_role is not None:
            chat.append({"role": self.developer_role, "content": self.developer_prompt})
        chat.append({"role": "user", "content": question})

        prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        return {"prompt": prompt, "item": item}


class CustomGRPOTrainer(GRPOTrainer):
    def __init__(
        self,
        model,
        args: GRPOConfig,
        tokenizer,
        train_dataset: ReasoningGymDataset,
        eval_dataset: ReasoningGymDataset,
    ):
        super().__init__(
            model=model,
            reward_funcs=[
                self._accuracy_reward,
                self._format_reward,
            ],
            args=args,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

    def _accuracy_reward(self, completions: list[str], **kwargs) -> list[float]:
        assert "item" in kwargs, "The 'item' argument must be provided to compute accuracy reward."
        assert len(kwargs["item"]) == len(completions), "Items and completions must have the same length."
        assert all(isinstance(item, dict) for item in kwargs["item"]), "Each item must be a dictionary."
        answers = [extract_answer(c) for c in completions]
        return [self.train_dataset.data.score_answer(answer, item) for answer, item in zip(answers, kwargs["item"])]

    def _format_reward(self, completions: list[str], **kwargs) -> list[float]:
        def count_tags(text: str) -> float:
            count = 0.0
            if re.search(r"\s*<think>\s*", text):
                count += 0.25
            if re.search(r"\s*</think>\s*", text):
                count += 0.25
            if re.search(r"\s*<answer>\s*", text):
                count += 0.25
            if re.search(r"\s*</answer>\s*", text):
                count += 0.25
            return count

        return [count_tags(c) for c in completions]


def make_dataset(
    tokenizer,
    data_source: Experiment | ProceduralDataset,
    developer_prompt: str,
    developer_role: Optional[str] = None,
) -> ReasoningGymDataset:
    """Create a ReasoningGymDataset from an Experiment or ProceduralDataset."""
    if isinstance(data_source, Experiment):
        return ReasoningGymDataset(
            tokenizer=tokenizer,
            experiment=data_source,
            developer_prompt=developer_prompt,
            developer_role=developer_role,
        )
    else:
        return ReasoningGymDataset(
            tokenizer=tokenizer,
            procedural_dataset=data_source,
            developer_prompt=developer_prompt,
            developer_role=developer_role,
        )


def prepare_datasets(
    config: DatasetConfig,
    tokenizer,
) -> tuple[ReasoningGymDataset, ReasoningGymDataset]:
    """Prepare the training and eval datasets."""
    developer_prompt = SYSTEM_PROMPTS[config.developer_prompt]

    dataset_specs = [
        DatasetSpec(
            name=name,
            weight=ds_config.weight,
            config=ds_config.config,
        )
        for name, ds_config in config.datasets.items()
    ]
    train_data_source = reasoning_gym.create_dataset(
        "composite", seed=1, size=config.dataset_size, datasets=dataset_specs
    )
    val_data_source = reasoning_gym.create_dataset(
        "composite", seed=2, size=config.dataset_size, datasets=dataset_specs
    )
    train_dataset = make_dataset(
        tokenizer=tokenizer,
        data_source=train_data_source,
        developer_prompt=developer_prompt,
        developer_role=config.developer_role,
    )
    eval_dataset = make_dataset(
        tokenizer=tokenizer,
        data_source=val_data_source,
        developer_prompt=developer_prompt,
        developer_role=config.developer_role,
    )
    return train_dataset, eval_dataset


def main():
    # -----------
    # Parse args
    # -----------
    parser = TrlParser((DatasetConfig, GRPOConfig, ModelConfig))
    reasoning_gym_args, training_args, model_args = parser.parse_args_and_config()
    set_seed(training_args.seed)

    # ---------------
    # Set up logging
    # ---------------
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Training parameters {training_args}")

    # -----------
    # Load model
    # -----------
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=model_args.attn_implementation,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    # --------------------
    # Instantiate trainer
    # --------------------
    training_args.reasoning_gym = reasoning_gym_args
    train_dataset, eval_dataset = prepare_datasets(reasoning_gym_args, tokenizer)
    trainer = CustomGRPOTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # ------------------------------
    # See if we can resume training
    # ------------------------------
    logger.info("Starting training...")
    # Check for last checkpoint
    ckpt = None
    if training_args.resume_from_checkpoint is not None:
        ckpt = training_args.resume_from_checkpoint
    elif os.path.isdir(training_args.output_dir):
        ckpt = get_last_checkpoint(training_args.output_dir)
    if ckpt:
        logger.info(f"\nCheckpoint detected, resuming training at {ckpt=}.")
    else:
        logger.info("\nNo checkpoint detected, starting training from scratch.")

    # ---------------
    # Start training
    # ---------------
    train_result = trainer.train(resume_from_checkpoint=ckpt)
    train_metrics = train_result.metrics
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    trainer.save_state()

    # ---------
    # Clean up
    # ---------
    del trainer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
