# Adapted version of Bytedance code:
# https://github.com/volcengine/verl/blob/a65c9157bc0b85b64cd753de19f94e80a11bd871/verl/trainer/main_ppo.py

import re

import re

import torch
from omegaconf import OmegaConf, open_dict
from torchdata.stateful_dataloader import StatefulDataLoader
from utils import ReasoningGymDataset
from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.dataset.rl_dataset import collate_fn

from reasoning_gym.utils import extract_answer


class RayGRPOTrainer(RayPPOTrainer):
    def __init__(
        self,
        config,
        tokenizer,
        train_dataset: ReasoningGymDataset,
        val_dataset: ReasoningGymDataset,
        role_worker_mapping: dict,
        resource_pool_manager,
        ray_worker_group_cls,
        max_output_length: int = 1024,
    ):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.max_output_length = max_output_length

        self.format_reward_scaling_factor = config.reward.format_reward.scaling_factor
        self.length_reward_scaling_factor = config.reward.length_reward.scaling_factor

        train_reward_fn = lambda data: self._score_output(data, num_examine=0)
        val_reward_fn = lambda data: self._score_output(data, num_examine=1)

        super().__init__(
            config,
            tokenizer,
            role_worker_mapping,
            resource_pool_manager,
            ray_worker_group_cls,
            train_reward_fn,
            val_reward_fn,
        )

    def _score_output(self, data: DataProto, num_examine: int = 0) -> torch.Tensor:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        num_printed = 0
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]  # tokenized prompts
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            response_str = self.tokenizer.decode(valid_response_ids)
            sequences_str = prompt_str + response_str

            index = data_item.non_tensor_batch["index"]

            reward = score = self._compute_correctness_score(
                solution_str=response_str,
                index=index,
            )

            if self.config.reward.format_reward.enable:
                format_reward = self._compute_format_reward(response_str)
                reward += format_reward
            else:
                format_reward = 0.0

            if self.config.reward.length_reward.enable:
                length_reward = self._compute_length_reward(response_str, score)
                reward += length_reward
            else:
                length_reward = 0.0

            reward_tensor[i, valid_response_length - 1] = reward

            if num_printed < num_examine:
                print(
                    f"reward={reward} (score={score}, format={format_reward}, length={length_reward}), seq={sequences_str}"
                )
                num_printed += 1

        return reward_tensor

    def _format_reward(solution_str: str):
        """Reward use of <think> and <answer> tags."""
        pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
        match = re.match(pattern, solution_str, re.DOTALL | re.MULTILINE)
        # Penalise if there are multiple <think> or <answer> tags
        if match is None:
            return 0
        if any(solution_str.count(tag) > 1 for tag in ["<think>", "<answer>", "</think>", "</answer>"]):
            return 0
        return 1

    def _compute_score(self, solution_str: str, index: int) -> float:
        found_answer = extract_answer(solution_str, tag_name="answer")
        data = self.train_dataset.data
        entry = data[index]
        if self.train_dataset.experiment:
            experiment = self.train_dataset.experiment
            return experiment.score_answer_with_id(found_answer, entry["metadata"]["entry_id"])
        else:
            return data.score_answer(found_answer, entry=entry)

    def _create_dataloader(self):
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.train_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
        )

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=len(self.val_dataset),
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        print(f"Size of train dataloader: {len(self.train_dataloader)}")
        print(f"Size of val dataloader: {len(self.val_dataloader)}")

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps
