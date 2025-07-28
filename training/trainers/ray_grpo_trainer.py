# Adapted version of Bytedance code:
# https://github.com/volcengine/verl/blob/a65c9157bc0b85b64cd753de19f94e80a11bd871/verl/trainer/main_ppo.py

import gc
import uuid
from copy import deepcopy

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from rewards import reward_registry
from torchdata.stateful_dataloader import StatefulDataLoader
from utils import ReasoningGymDataset
from verl import DataProto
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    _timer,
    apply_kl_penalty,
    compute_advantage,
    compute_data_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
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

        if config.curriculum.enabled:
            self.last_k = config.curriculum.last_k
        else:
            self.last_k = None

        self.reward_functions = []
        if hasattr(config, "reward") and hasattr(config.reward, "secondary_rewards"):
            for func_config in config.reward.secondary_rewards:
                func_name = func_config.name
                scaling_factor = func_config.get("scaling_factor", 1.0)
                func = reward_registry.get(func_name)
                if func:
                    # Store both function and its arguments
                    self.reward_functions.append(
                        {
                            "function": func,
                            "name": func_name,
                            "scaling_factor": scaling_factor,
                            "kwargs": func_config.get("kwargs", {}),
                        }
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
            correctness_score = self._compute_correctness_score(
                solution_str=response_str,
                index=index,
            )
            if self.config.reward.use_accuracy:
                reward_components = {"correctness": correctness_score}
                total_reward = correctness_score
            else:
                reward_components = {}
                total_reward = 0

            for reward_fn in self.reward_functions:
                func = reward_fn["function"]
                name = reward_fn["name"]
                scaling_factor = reward_fn["scaling_factor"]
                kwargs = reward_fn["kwargs"]
                if name == "cosine":
                    is_correct = correctness_score == 1.0
                    reward = func(response_str, scaling_factor, is_correct=is_correct, **kwargs)
                elif name == "length":
                    reward = func(response_str, scaling_factor, correctness_score=correctness_score, **kwargs)
                else:
                    reward = func(response_str, scaling_factor, **kwargs)
                reward_components[name] = reward
                total_reward += reward

            reward_tensor[i, valid_response_length - 1] = total_reward

            if num_printed < num_examine:
                components = ", ".join([f"{k}={v:.2f}" for k, v in reward_components.items()])
                print(f"(score={total_reward}, seq={sequences_str}, response={response_str})")
                print(f"reward={total_reward:.2f} ({components})")
                num_printed += 1

        return reward_tensor

    def _compute_correctness_score(self, solution_str: str, index: int) -> float:
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

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            print(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                if "multi_modal_inputs" in batch.non_tensor_batch.keys():
                    gen_batch = batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "multi_modal_inputs"],
                    )
                else:
                    gen_batch = batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_tensor = self.reward_fn(batch)
                        batch.batch["token_level_scores"] = reward_tensor

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.get("use_kl_loss", False):
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                        )

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # collect metrics
                if self.config.curriculum.enabled:
                    grouped_scores = self.train_dataset.aggregate(last_n=self.config.curriculum.last_k)

                    if self.config.curriculum.schedule.automatic:
                        for dataset_name in grouped_scores.keys():
                            if self.global_steps % self.config.curriculum.schedule.update_steps == 0:
                                self.train_dataset.update_experiment_difficulty(dataset_name, method="increment")
                    else:
                        for dataset_name in grouped_scores.keys():
                            if (
                                grouped_scores[dataset_name]["results"] > self.config.curriculum.success_threshold
                            ) and (grouped_scores[dataset_name]["total_samples"] >= self.config.curriculum.last_k):
                                print(
                                    f"Increasing difficulty for dataset: {dataset_name} (success rate: {grouped_scores[dataset_name]['results']:.2f}, samples: {grouped_scores[dataset_name]['total_samples']})"
                                )
                                self.train_dataset.update_experiment_difficulty(dataset_name, method="increment")
                            elif (
                                grouped_scores[dataset_name]["results"] < self.config.curriculum.failure_threshold
                            ) and (grouped_scores[dataset_name]["total_samples"] >= self.config.curriculum.last_k):
                                print(
                                    f"Decreasing difficulty for dataset: {dataset_name} (success rate: {grouped_scores[dataset_name]['results']:.2f}, samples: {grouped_scores[dataset_name]['total_samples']})"
                                )
                                self.train_dataset.update_experiment_difficulty(dataset_name, method="decrement")

                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    print(f"Final validation metrics: {last_val_metrics}")
                    return

                self.global_steps += 1
                gc.collect()
