# Adapted version of Bytedance code:
# https://github.com/volcengine/verl/blob/a65c9157bc0b85b64cd753de19f94e80a11bd871/verl/trainer/main_ppo.py

import re
import uuid

import torch
import numpy as np
from omegaconf import OmegaConf, open_dict
from torchdata.stateful_dataloader import StatefulDataLoader
from utils import ReasoningGymDataset
from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_advantage, apply_kl_penalty, _timer, compute_data_metrics, compute_throughout_metrics, compute_timing_metrics, reduce_metrics
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
        if config.curriculum.enabled:
            self.last_k = config.curriculum.last_k
        else:
            self.last_k = None

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

            score = self._compute_correctness_score(
                solution_str=response_str,
                index=index,
            )
            format_reward = self._compute_format_reward(response_str)

            reward = score + format_reward 

            reward_tensor[i, valid_response_length - 1] = reward

            if num_printed < num_examine:
                print(
                    f"reward={reward} (score={score}, format={format_reward}, seq={sequences_str}"
                )
                num_printed += 1

        return reward_tensor

    def _compute_format_reward(self, solution_str: str) -> float:
        """Reward use of exactly one correctly structured <think> and <answer> block."""
        scaling_factor = self.format_reward_scaling_factor
        # check <think> and <answer> blocks are present
        pattern = r"\s*<think>.*?</think>\s*<answer>.*?</answer>"
        if not re.match(pattern, solution_str, re.DOTALL):
            return 0.0
        # check exactly one properly structured <think> block and one <answer> block
        think_matches = list(re.finditer(r"<think>(.*?)</think>", solution_str, re.DOTALL))
        answer_matches = list(re.finditer(r"<answer>(.*?)</answer>", solution_str, re.DOTALL))
        if len(think_matches) != 1 or len(answer_matches) != 1:
            return 0.0
        # check for <think> or <answer> inside <think>
        think_content = think_matches[0].group(1)
        if "<think>" in think_content or "<answer>" in think_content:
            return 0.0
        # check for nested <think> or <answer> inside <answer>
        answer_content = answer_matches[0].group(1)
        if "<answer>" in answer_content or "<think>" in answer_content:
            return 0.0
        return 1.0 * scaling_factor

    def _compute_length_reward(
        self,
        solution_str: str,
        correctness_score: float,
        max_score: float = 1.0,
    ) -> float:
        """
        Reward shorter solutions for perfect answers, longer solutions for imperfect answers.
        The scaling factor for this should be set far below 1.0, to avoid dominating the reward signal over correctness.
        """
        epsilon = 1e-6
        scaling_factor = self.length_reward_scaling_factor
        generation_len = len(solution_str)
        progress = min(generation_len / self.max_output_length, 1.0)
        if correctness_score < max_score - epsilon:
            # for imperfect answers, incentivise longer ones
            length_reward = (max_score - correctness_score) * progress
        else:
            # for perfect answers, penalise longer ones
            length_reward = -progress
        return length_reward * scaling_factor

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
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            print(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                
                gen_batch = batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids'],
                    )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer('step', timing_raw):
                    # generate a batch
                    with _timer('gen', timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                             dtype=object)
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # recompute old_log_probs
                    with _timer('old_log_prob', timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    with _timer('adv', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_tensor = self.reward_fn(batch)
                        batch.batch['token_level_scores'] = reward_tensor

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        (is_last_step or  self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and ( is_last_step or \
                            self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                if type(self.train_dataset.experiment):
                    grouped_scores = self.train_dataset.aggregate(last_n=self.last_k)
                    for dataset_name in grouped_scores.keys():
                        if (grouped_scores[dataset_name]['results'] > self.success_threshold) and (grouped_scores[dataset_name]['total_samples'] > self.last_k):
                            self.train_dataset.experiment.update_difficulty(dataset_name, method='increment')
                        elif (grouped_scores[dataset_name]['results'] < self.failure_threshold) and (grouped_scores[dataset_name]['total_samples'] > self.last_k):
                            self.train_dataset.update_difficulty(dataset_name, method='decrement')
                    
               
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    print(f'Final validation metrics: {last_val_metrics}')
                    return

                self.global_steps += 1