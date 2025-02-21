# This example is an adapted version of Bytedance's code:
# https://github.com/volcengine/verl/blob/a65c9157bc0b85b64cd753de19f94e80a11bd871/verl/trainer/main_ppo.py
import os
from typing import Optional

import hydra
import ray
import torch
import verl.utils.torch_functional as verl_F
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer
from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask

import reasoning_gym
import reasoning_gym.utils
from reasoning_gym.utils import extract_answer
from tools.server.models import AnswerItem, BatchEntry, ExperimentCreate


class ReasoningGymDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        dataset_name: str,
        seed: int,
        size: int,
        developer_prompt: Optional[str] = None,
        developer_role: str = "system",
        max_prompt_length: int = 2048,
        truncation: str = "error",  ##  ['left', 'right', 'error']
        return_raw_chat: bool = False,
        server_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        batch_size: int = 32,
    ):
        from tools.cli.rgc.client import RGClient

        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.developer_prompt = developer_prompt
        self.developer_role = developer_role
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.return_raw_chat = return_raw_chat
        self.size = size
        self.batch_size = batch_size

        # Initialize client and create experiment if needed
        self.client = RGClient(base_url=server_url, api_key=api_key)

        # Check if experiment exists, create if not
        experiments = self.client.list_experiments()
        if dataset_name not in experiments.experiments:
            config = ExperimentCreate(
                name=dataset_name,
                size=size,
                seed=seed,
                datasets={dataset_name: {"weight": 1.0, "config": {"seed": seed, "size": size}}},
            )
            self.client.create_experiment(dataset_name, config)

        # Cache for batches
        self._batch_cache: dict[int, list[BatchEntry]] = {}

    def __len__(self) -> int:
        return self.size

    def _get_batch(self, batch_idx: int) -> list[BatchEntry]:
        """Fetch or retrieve cached batch"""
        if batch_idx not in self._batch_cache:
            base_index = batch_idx * self.batch_size
            response = self.client.get_batch(self.dataset_name, base_index=base_index, batch_size=self.batch_size)
            self._batch_cache[batch_idx] = response.entries

            # # Basic cache management - keep only last N batches
            # if len(self._batch_cache) > 10:
            #     oldest_batch = min(self._batch_cache.keys())
            #     del self._batch_cache[oldest_batch]

        return self._batch_cache[batch_idx]

    def __getitem__(self, index):
        # Get batch containing this index
        batch_idx = index // self.batch_size

        batch = self._get_batch(batch_idx)
        entry = batch[index % self.batch_size]

        # Format chat/prompt
        chat = []
        if self.developer_prompt is not None:
            chat.append({"role": self.developer_role, "content": self.developer_prompt})
        chat.append({"role": "user", "content": entry.question})

        prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        # Tokenize
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt,
            tokenizer=self.tokenizer,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict = {
            "data_source": "reasoning_gym/" + self.dataset_name,
            "input_ids": input_ids[0],
            "attention_mask": attention_mask[0],
            "position_ids": position_ids[0],
            "entry_id": entry.entry_id,
            "metadata": entry.metadata,
            "index": index,
        }

        # Add raw chat if requested
        if self.return_raw_chat:
            row_dict["raw_prompt"] = chat

        return row_dict


class RayPPOTrainerCustom(RayPPOTrainer):
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict,
        resource_pool_manager,
        ray_worker_group_cls,
        dataset_name: str = "chain_sum",
        dataset_size: int = 10000,
    ):
        self.dataset_name = dataset_name
        self.dataset_size = dataset_size

        developer_prompt = reasoning_gym.utils.SYSTEM_PROMPTS["DeepSeekZero"]
        rg_api_key = os.getenv("REASONING_GYM_API_KEY", "your-secret-key")
        self.train_dataset = ReasoningGymDataset(
            tokenizer=tokenizer,
            dataset_name=self.dataset_name,
            seed=1,
            size=self.dataset_size,
            developer_prompt=developer_prompt,
            api_key=rg_api_key,
        )

        self.val_dataset = ReasoningGymDataset(
            tokenizer=tokenizer,
            dataset_name=self.dataset_name,
            seed=2,
            size=self.dataset_size,
            developer_prompt=developer_prompt,
            api_key=rg_api_key,
        )

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

        # Prepare batch of answers to score
        answer_items = []
        valid_response_lengths = []
        sequences_strs = []

        for i in range(len(data)):
            data_item = data[i]

            # Get prompt and response
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            valid_response_lengths.append(valid_response_length)

            # Decode full sequence
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)
            sequences_strs.append(sequences_str)

            # Extract answer and prepare scoring item
            found_answer = extract_answer(sequences_str, tag_name="answer")

            index = data_item.non_tensor_batch["index"]
            entry_id = self.train_dataset[index]["entry_id"]
            # print(
            #     "found_answer",
            #     entry_id,
            #     found_answer,
            # )

            answer_items.append(AnswerItem(entry_id=entry_id, answer=found_answer))

        # Score all answers in one request
        response = self.train_dataset.client.score_outputs(self.train_dataset.dataset_name, answer_items)
        # print("response", response)

        # Fill reward tensor
        for i, (score, valid_response_length) in enumerate(zip(response.scores, valid_response_lengths)):
            reward_tensor[i, valid_response_length - 1] = score

            if i < num_examine:
                print(f"reward={score}, seq={sequences_strs[i]}")

        return reward_tensor

    def _create_dataloader(self):
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.train_batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=collate_fn,
        )

        self.val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=len(self.val_dataset),
            shuffle=False,
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


@ray.remote
def main_task(config):
    # print initial config
    from pprint import pprint

    from verl.utils import hf_tokenizer
    from verl.utils.fs import copy_local_path_from_hdfs

    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == "fsdp":
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.single_controller.ray import RayWorkerGroup
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker

        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == "megatron":
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayPPOTrainerCustom(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
    )
    trainer.init_workers()
    trainer.fit()


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}})

    ray.get(main_task.remote(config))


if __name__ == "__main__":
    main()
