# tested with OpenRLHF  707b970e992154952a91607ca5491cc49b8665c3

import argparse
import itertools
import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
from openrlhf.datasets import PromptDataset, SFTDataset
from openrlhf.models import Actor, get_llm_for_sequence_regression
from openrlhf.models.utils import compute_approx_kl, masked_mean
from openrlhf.trainer import PPOTrainer
from openrlhf.trainer.ppo_utils.experience_maker import Experience, NaiveExperienceMaker, Samples
from openrlhf.utils import blending_datasets, get_strategy, get_tokenizer
from openrlhf.utils.logging_utils import init_logger
from torch.utils.data import Dataset
from transformers.trainer import get_scheduler

import reasoning_gym
from reasoning_gym.dataset import ProceduralDataset
from reasoning_gym.utils import extract_answer

logger = init_logger(__name__)

DEBUG = False


def preprocess_data(data, input_template=None, input_key="input", apply_chat_template=None) -> str:
    if apply_chat_template:
        chat = data[input_key]
        if isinstance(chat, str):
            chat = [{"role": "user", "content": chat}]
        prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    else:
        prompt = data[input_key]
        if input_template:
            prompt = input_template.format(prompt)
    return prompt


class ReasoningGymDataset(Dataset):
    def __init__(
        self,
        dataset,
        tokenizer,
        developer_prompt: Optional[str] = None,
        developer_role: str = "system",
    ) -> None:
        super().__init__()

        self.dataset = dataset
        self.tokenizer = tokenizer
        self.developer_prompt = developer_prompt
        self.developer_role = developer_role

    def __len__(self):
        length = len(self.dataset)
        return length

    def __getitem__(self, idx: int) -> tuple[str, dict]:
        x = self.dataset[idx]

        q = x["question"]
        chat = []
        if self.developer_prompt is not None:
            chat.append({"role": self.developer_role, "content": self.developer_prompt})
        chat.append({"role": "user", "content": q})
        prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        return prompt, x


@dataclass
class SamplesWithMetadata(Samples):
    metadata: Optional[dict]


class AlgorithmicRewardExperienceMaker(NaiveExperienceMaker):
    def __init__(
        self,
        dataset: ProceduralDataset,
        actor: Actor,
        critic: nn.Module,
        reward_model: nn.Module,
        initial_model: Actor,
        tokenizer,
        prompt_max_len: int,
        kl_controller,
        strategy=None,
        remote_rm_url: str = None,
        reward_fn=None,
    ) -> None:
        super().__init__(
            actor=actor,
            critic=critic,
            reward_model=reward_model,
            initial_model=initial_model,
            tokenizer=tokenizer,
            prompt_max_len=prompt_max_len,
            kl_controller=kl_controller,
            strategy=strategy,
            remote_rm_url=remote_rm_url,
            reward_fn=reward_fn,
        )
        self.dataset = dataset

    @torch.no_grad()
    def generate_samples(self, all_prompts: List[Tuple[str, Any]], **generate_kwargs) -> List[Samples]:
        """
        Generate samples and return in batches.
        """
        assert not getattr(self, "packing_samples", False)
        args = self.strategy.args
        self.actor.eval()

        # prepare inputs to sample multiple response
        repeated_prompts = []
        repeated_metadata = []
        for prompt, metadata in all_prompts:
            for _ in range(args.n_samples_per_prompt):
                repeated_prompts.append(prompt)
                repeated_metadata.append(metadata)

        samples_list = []
        for i in range(0, len(repeated_prompts), args.micro_rollout_batch_size):
            prompts = repeated_prompts[i : i + args.micro_rollout_batch_size]
            metadata = repeated_metadata[i : i + args.micro_rollout_batch_size]
            inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
            sequences, attention_mask, action_mask = self.actor.generate(**inputs, **generate_kwargs)
            samples = SamplesWithMetadata(
                sequences=sequences,
                attention_mask=attention_mask,
                action_mask=action_mask,
                num_actions=action_mask.size(1),
                packed_seq_lens=None,
                response_length=action_mask.float().sum(dim=-1),
                total_length=attention_mask.float().sum(dim=-1),
                metadata=metadata,
            )
            samples_list.append(samples)
        return samples_list

    @torch.no_grad()
    def make_experience(self, samples: Samples) -> Experience:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        self.actor.eval()
        self.initial_model.eval()
        if self.reward_model is not None:
            self.reward_model.eval()
        if self.critic is not None:
            self.critic.eval()

        # extract values from samples
        sequences = samples.sequences
        attention_mask = samples.attention_mask
        action_mask = samples.action_mask
        num_actions = samples.num_actions
        if isinstance(samples, SamplesWithMetadata):
            metadata = samples.metadata
        else:
            metadata = None

        # log probs
        action_log_probs = self.actor(sequences, num_actions, attention_mask)

        # init log probs
        base_action_log_probs = self.initial_model(sequences, num_actions, attention_mask)

        # values
        if self.critic is not None:
            value = self.critic(sequences, num_actions, attention_mask)
        else:
            value = None

        # determine outcome reward
        completions = sequences[:, -action_mask.size(1) :].cpu()
        completions = self.tokenizer.batch_decode(completions, skip_special_tokens=True)
        returns = [
            self.dataset.score_answer(extract_answer(c, tag_name="answer"), entry=m)
            for c, m in zip(completions, metadata)
        ]
        r = torch.tensor(returns, dtype=torch.float32, device=sequences.device)

        kl = compute_approx_kl(
            action_log_probs,
            base_action_log_probs,
            action_mask=action_mask,
            use_kl_estimator_k3=self.strategy.args.use_kl_estimator_k3,
        )

        info = {
            "kl": masked_mean(kl, action_mask, dim=-1),
            "reward": r,
            "response_length": samples.response_length,
            "total_length": samples.total_length,
            "num_actions": num_actions,
        }

        logger.info(f"info={info}")

        # reset model state
        self.actor.train()
        if self.critic is not None:
            self.critic.train()

        return Experience(
            sequences,
            action_log_probs,
            value,
            None,
            None,
            attention_mask,
            action_mask,
            info,
            kl,
        )


def train(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()

    # configure model
    # load huggingface model
    actor = Actor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        ds_config=strategy.get_ds_train_config(is_actor=True),
    )

    if args.actor_init_on_gpu:
        actor = actor.to(torch.cuda.current_device())

    if args.critic_pretrain:
        critic = get_llm_for_sequence_regression(
            args.critic_pretrain,
            "critic",
            normalize_reward=args.normalize_reward,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=args.target_modules,
            lora_dropout=args.lora_dropout,
            ds_config=strategy.get_ds_train_config(is_actor=False),
            value_head_prefix=args.value_head_prefix,
            init_value_head=strategy.args.pretrain == strategy.args.critic_pretrain,
        )
    else:
        critic = None

    reward_model = None

    strategy.print(actor)
    strategy.print(critic)

    # configure tokenizer
    tokenizer = get_tokenizer(
        args.pretrain,
        actor.model,
        "left",
        strategy,
        use_fast=not args.disable_fast_tokenizer,
    )

    # load weights for reference actor
    initial_model = Actor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        ds_config=strategy.get_ds_eval_config(offload=False),
    )

    if args.enable_ema:
        ema_model = Actor(
            args.pretrain,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            ds_config=strategy.get_ds_eval_config(offload=True),
        )
    else:
        ema_model = None

    # gradient_checkpointing
    if args.gradient_checkpointing:
        actor.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )
        if critic is not None:
            critic.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
            )

    # configure optimizer
    actor_optim = strategy.create_optimizer(
        actor, lr=args.actor_learning_rate, betas=args.adam_betas, weight_decay=args.l2
    )
    if args.critic_pretrain:
        critic_optim = strategy.create_optimizer(
            critic,
            lr=args.critic_learning_rate,
            betas=args.adam_betas,
            weight_decay=args.l2,
        )
    else:
        critic_optim = None

    # prepare datasets
    print("prompt_data", args.prompt_data)

    # DeepSeek Zero system prompt
    system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>
"""

    prompts_data = reasoning_gym.create_dataset(args.prompt_data, size=args.max_samples)
    prompts_dataset = ReasoningGymDataset(prompts_data, tokenizer, developer_prompt=system_prompt)

    if args.pretrain_data:
        pretrain_data = blending_datasets(
            args.pretrain_data,
            args.pretrain_data_probs,
            strategy,
            args.seed,
            return_eval=False,
            train_split=args.pretrain_split,
        )
        pretrain_max_len = args.max_len if args.max_len else args.prompt_max_len + args.generate_max_len
        pretrain_dataset = SFTDataset(
            pretrain_data.select(
                range(
                    min(
                        len(pretrain_data),
                        args.max_epochs * len(prompts_dataset) * args.n_samples_per_prompt,
                    )
                )
            ),
            tokenizer,
            pretrain_max_len,
            strategy,
            pretrain_mode=True,
        )

    def collate_prompt_and_metadata(xs: list[tuple]) -> list[tuple]:
        """dummy collate function to pass on the metadata dict unchanged"""
        return xs

    # prepare dataloader
    prompts_dataloader = strategy.setup_dataloader(
        prompts_dataset,
        args.rollout_batch_size // strategy.world_size,
        True,
        True,
        collate_fn=collate_prompt_and_metadata,
    )
    if args.pretrain_data:
        pretrain_dataloader = itertools.cycle(
            iter(
                strategy.setup_dataloader(
                    pretrain_dataset,
                    args.micro_train_batch_size,
                    True,
                    True,
                    pretrain_dataset.collate_fn,
                )
            )
        )
    else:
        pretrain_dataloader = None

    # configure scheduler
    num_update_steps_per_episodes = (
        len(prompts_dataset) * args.n_samples_per_prompt // args.train_batch_size * args.max_epochs
    )
    max_steps = math.ceil(args.num_episodes * num_update_steps_per_episodes)

    actor_scheduler = get_scheduler(
        "cosine_with_min_lr",
        actor_optim,
        num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
        num_training_steps=max_steps,
        scheduler_specific_kwargs={"min_lr": args.actor_learning_rate * 0.1},
    )

    if args.critic_pretrain:
        critic_scheduler = get_scheduler(
            "cosine_with_min_lr",
            critic_optim,
            num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
            num_training_steps=max_steps,
            scheduler_specific_kwargs={"min_lr": args.critic_learning_rate * 0.1},
        )
    else:
        critic_scheduler = None

    # prepare models/optimizers...
    (
        (actor, actor_optim, actor_scheduler),
        (critic, critic_optim, critic_scheduler),
        reward_model,
        initial_model,
    ) = strategy.prepare(
        (actor, actor_optim, actor_scheduler),
        (critic, critic_optim, critic_scheduler),
        reward_model,
        initial_model,
        is_rlhf=True,
    )

    if ema_model:
        ema_model._offload = True
        ema_model = strategy.prepare(ema_model, is_rlhf=True)

    # load checkpoint
    consumed_samples = 0
    if args.load_checkpoint and os.path.exists(os.path.join(args.ckpt_path, "_actor")):
        _, states = strategy.load_ckpt(actor.model, os.path.join(args.ckpt_path, "_actor"))
        if args.critic_pretrain:
            strategy.load_ckpt(critic, os.path.join(args.ckpt_path, "_critic"))
        consumed_samples = states["consumed_samples"]
        strategy.print(f"Loaded the checkpoint: {args.ckpt_path}, consumed_samples: {consumed_samples}")

    os.makedirs(args.save_path, exist_ok=True)

    # configure Trainer
    trainer = PPOTrainer(
        strategy,
        actor,
        critic,
        reward_model,
        initial_model,
        ema_model,
        actor_optim,
        critic_optim,
        actor_scheduler,
        critic_scheduler,
        max_epochs=args.max_epochs,
        micro_train_batch_size=args.micro_train_batch_size,
        micro_rollout_batch_size=args.micro_rollout_batch_size,
        gradient_checkpointing=args.gradient_checkpointing,
        tokenizer=tokenizer,
        prompt_max_len=args.prompt_max_len,
        value_clip=args.value_clip,
        eps_clip=args.eps_clip,
        gamma=args.gamma,
        lambd=args.lambd,
        init_kl_coef=args.init_kl_coef,
        kl_target=args.kl_target,
        ema_beta=0.992,
        ptx_coef=args.ptx_coef,
        max_norm=args.max_norm,
        # fro GPT generation
        do_sample=True,
        max_new_tokens=args.generate_max_len,
        max_length=args.max_len,
        temperature=args.temperature,
        top_p=args.top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        # remote reward model
        remote_rm_url=args.remote_rm_url,
        save_hf_ckpt=args.save_hf_ckpt,
        disable_ds_ckpt=args.disable_ds_ckpt,
    )

    # patch experience maker ..
    xp = trainer.experience_maker
    trainer.experience_maker = AlgorithmicRewardExperienceMaker(
        dataset=prompts_dataset.dataset,
        actor=xp.actor,
        critic=xp.critic,
        reward_model=xp.reward_model,
        initial_model=xp.initial_model,
        tokenizer=xp.tokenizer,
        prompt_max_len=xp.prompt_max_len,
        kl_controller=xp.kl_ctl,
        strategy=xp.strategy,
        remote_rm_url=xp.remote_rm_url,
        reward_fn=xp.reward_fn,
    )
    xp = None

    trainer.fit(
        args,
        prompts_dataloader,
        pretrain_dataloader,
        consumed_samples,
        num_update_steps_per_episodes,
    )

    # save model checkpoint after fitting on only rank0
    strategy.save_model(
        ema_model if args.enable_ema else actor,
        tokenizer,
        args.save_path,
    )

    if args.critic_pretrain and args.save_value_network:
        strategy.save_model(
            critic,
            tokenizer,
            args.save_path + "_critic",
        )


import os

import debugpy

if __name__ == "__main__":

    # Only attach debugger on rank 0 process
    if DEBUG and int(os.getenv("LOCAL_RANK", "0")) == 0:
        debugpy.listen(("localhost", 5678))
        print("Waiting for debugger attach on localhost:5678")
        debugpy.wait_for_client()

    parser = argparse.ArgumentParser()
    # Checkpoint
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--save_hf_ckpt", action="store_true", default=False)
    parser.add_argument("--disable_ds_ckpt", action="store_true", default=False)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_ppo")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1e8)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)

    # PPO
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--rollout_batch_size", type=int, default=512)
    parser.add_argument("--micro_rollout_batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--prompt_max_len", type=int, default=1024, help="Max tokens for each prompt")
    parser.add_argument(
        "--generate_max_len",
        type=int,
        default=1024,
        help="Max tokens to generate in PPO",
    )
    parser.add_argument("--max_len", type=int, default=None, help="deprecated max_len")
    parser.add_argument("--max_samples", type=int, default=1000000)
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--l2", type=float, default=0.0, help="weight decay loss")
    parser.add_argument("--ptx_coef", type=float, default=0.05, help="PPO-ptx loss coef")
    parser.add_argument("--eps_clip", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--value_clip", type=float, default=0.2, help="PPO value clip range")
    parser.add_argument("--lambd", type=float, default=1.0, help="PPO GAE lambd")
    parser.add_argument("--gamma", type=float, default=1, help="PPO GAE gamma")
    parser.add_argument("--micro_train_batch_size", type=int, default=4, help="batch size per GPU")
    parser.add_argument("--train_batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument(
        "--normalize_reward",
        action="store_true",
        default=False,
        help="Enable Reward Normazation",
    )
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--freezing_actor_steps",
        type=int,
        default=-1,
        help="Used for critic initialization",
    )
    parser.add_argument(
        "--n_samples_per_prompt",
        type=int,
        default=1,
        help="number of responses for each prompt in generation",
    )
    parser.add_argument(
        "--save_value_network",
        action="store_true",
        default=False,
        help="Save critic model",
    )
    parser.add_argument("--actor_learning_rate", type=float, default=1e-6)
    parser.add_argument("--critic_learning_rate", type=float, default=9e-6)
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.03)
    parser.add_argument("--kl_target", type=float, default=None)
    parser.add_argument("--init_kl_coef", type=float, default=0.01, help="KL penalty in PPO")
    parser.add_argument(
        "--use_kl_estimator_k3",
        action="store_true",
        default=False,
        help=(
            "Use the k3 estimator in http://joschu.net/blog/kl-approx.html"
            "to ensure the KL divergence calculated is non-negative"
        ),
    )
    parser.add_argument(
        "--adam_betas",
        type=float,
        nargs=2,
        default=(0.9, 0.95),
        help="Betas for Adam optimizer",
    )
    parser.add_argument(
        "--reward_clip_range",
        type=float,
        nargs=2,
        default=(-10, 10),
        help="Reward clip range",
    )

    # DeepSpeed
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--enable_ema", action="store_true", help="Enable EMA checkpoint for the model.")
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument(
        "--adam_offload",
        action="store_true",
        default=False,
        help="Offload Adam Optimizer",
    )
    parser.add_argument("--actor_init_on_gpu", action="store_true", default=False)
    parser.add_argument(
        "--flash_attn",
        action="store_true",
        default=False,
        help="Enable FlashAttention2",
    )
    parser.add_argument("--aux_loss_coef", type=float, default=0, help="MoE balancing loss")
    parser.add_argument("--grad_accum_dtype", type=str, default=None, help="Adam grad accum data type")
    parser.add_argument("--overlap_comm", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)

    # Reinforce
    parser.add_argument(
        "--advantage_estimator",
        type=str,
        choices=["gae", "reinforce", "rloo"],
        default="gae",
        help="Choose advantage estimation method: gae, reinforce, rloo",
    )

    # LoRA
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--lora_dropout", type=float, default=0)

    # Models
    parser.add_argument("--pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--reward_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--remote_rm_url", type=str, default=None, help="remote RM API")
    parser.add_argument("--critic_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--value_head_prefix", type=str, default="score")

    # Custom dataset
    parser.add_argument("--prompt_data", type=str, default="chain_sum", help="HF dataset name or path")
    parser.add_argument(
        "--prompt_data_probs",
        type=str,
        default="1.0",
        help="sampling probs for datasets",
    )
    parser.add_argument("--prompt_split", type=str, default="train")
    parser.add_argument("--pretrain_data", type=str, default=None, help="HF dataset name or path")
    parser.add_argument(
        "--pretrain_data_probs",
        type=str,
        default="1.0",
        help="sampling probs for datasets",
    )
    parser.add_argument("--pretrain_split", type=str, default="train")
    parser.add_argument("--input_key", type=str, default="question", help="JSON dataset key")
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        default=False,
        help="Use HF tokenizer chat template",
    )

    # wandb parameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_ppo")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="ppo_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    # TensorBoard parameters
    parser.add_argument("--use_tensorboard", type=str, default=None, help="TensorBoard logging path")

    args = parser.parse_args()

    if args.advantage_estimator not in ["gae"]:
        args.critic_pretrain = None
    elif args.critic_pretrain is None:
        args.critic_pretrain = args.pretrain  ## temp

    if args.advantage_estimator == "rloo":
        assert args.n_samples_per_prompt > 1, "RLOO requires n_samples_per_prompt > 1"

    if args.input_template and "{}" not in args.input_template:
        print("[Warning] {} not in args.input_template, set to None")
        args.input_template = None

    if args.input_template and "\\n" in args.input_template:
        print(
            "[Warning] input_template contains \\n chracters instead of newline. "
            "You likely want to pass $'\\n' in Bash or \"`n\" in PowerShell."
        )

    train(args)
