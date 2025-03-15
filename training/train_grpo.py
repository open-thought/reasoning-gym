"""Train an LLM using GRPO over Reasoning Gym procedural dataset(s)."""

import hydra
import ray
from omegaconf import OmegaConf
from trainers import RayGRPOTrainer
from utils import ReasoningGymDataset

import reasoning_gym
import reasoning_gym.utils
from reasoning_gym.composite import DatasetSpec


def prepare_datasets(config, tokenizer) -> tuple[ReasoningGymDataset, ReasoningGymDataset]:
    """Prepare training and validation datasets."""
    # TODO: load these values from config
    dataset_name = "composite"
    dataset_names = ["mini_sudoku", "futoshiki", "sudoku"]
    dataset_size = 10000
    developer_prompt = reasoning_gym.utils.SYSTEM_PROMPTS["DeepSeekZero"]

    if dataset_name == "composite":
        dataset_specs = [DatasetSpec(name=name, size=dataset_size, config={}) for name in dataset_names]
        train_procedural_dataset = reasoning_gym.create_dataset("composite", seed=1, datasets=dataset_specs)
        val_procedural_dataset = reasoning_gym.create_dataset("composite", seed=2, datasets=dataset_specs)
    else:
        train_procedural_dataset = reasoning_gym.create_dataset(dataset_name, seed=1, size=dataset_size)
        val_procedural_dataset = reasoning_gym.create_dataset(dataset_name, seed=2, size=dataset_size)

    train_dataset = ReasoningGymDataset(
        tokenizer=tokenizer,
        procedural_dataset=train_procedural_dataset,
        dataset_name=dataset_name,
        developer_prompt=developer_prompt,
    )

    val_dataset = ReasoningGymDataset(
        tokenizer=tokenizer,
        procedural_dataset=val_procedural_dataset,
        dataset_name=dataset_name,
        developer_prompt=developer_prompt,
    )

    return train_dataset, val_dataset


@ray.remote
def main_task(config):
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

    train_dataset, val_dataset = prepare_datasets(config, tokenizer)

    trainer = RayGRPOTrainer(
        config=config,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
    )
    trainer.init_workers()
    trainer.fit()


@hydra.main(config_path="configs", config_name="llama3.1_1b_grpo", version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}})
    ray.get(main_task.remote(config))


if __name__ == "__main__":
    main()
