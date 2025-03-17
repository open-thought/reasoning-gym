import os
from argparse import Namespace
import time
from dataclasses import asdict, dataclass
import json
import ray
from ray.util.placement_group import placement_group
import torch
import numpy as np
import pandas as pd
from transformers import PreTrainedTokenizer, AutoTokenizer
from vllm import SamplingParams
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import threading
import shutil
from typing import Callable, List
from queue import Queue

from open_instruct.grpo_fast import Timer, collate_fn, Args, calculate_runtime_args, ModelGroup, PolicyTrainerRayProcess, vllm_generate_thread
from open_instruct.ground_truth_utils import soft_format_reward_func
from open_instruct.model_utils import ModelConfig, push_folder_to_hub
from open_instruct.rl_utils2 import PackedSequences, reset_position_ids
from open_instruct.utils import get_wandb_tags, is_beaker_job, maybe_get_beaker_config, ArgumentParserPlus
from open_instruct.vllm_utils2 import create_vllm_engines


from reasoning_gym.utils import extract_answer, SYSTEM_PROMPTS

from utils import list_preserving_collate, pack_sequences, ReasoningGymDataset

@dataclass
class GRPOArgs(Args):
    dataset_name: str = "chain_sum"
    seed: int = 42
    size: int = 10000
    eval_seed: int = 42
    eval_size: int = 100
    
    

def data_preparation_thread(
    reward_fn: Callable,
    inference_results_Q: Queue,
    packed_sequences_Q: Queue,
    queries_prompt_Q: Queue,
    args: Args,
    tokenizer: PreTrainedTokenizer,
    num_training_steps: int,
    reasoning_gym_dataset: ReasoningGymDataset,
):
    for training_step in range(1, num_training_steps + 1):
        # Get next batch of prompts and responses
        observations = queries_prompt_Q.get()
        queries, items = observations
        
        if args.num_samples_per_prompt_rollout > 1:
            queries = [item for item in queries for _ in range(args.num_samples_per_prompt_rollout)]
        with Timer("🚀 [Data Preparation Thread] Getting response ids"):
            responses, finish_reasons = inference_results_Q.get()
            for i in range(len(finish_reasons)):
                if finish_reasons[i] == "stop" and responses[i][-1] != tokenizer.eos_token_id:
                    responses[i].append(tokenizer.eos_token_id)

        with Timer("📦 [Data Preparation Thread] Packing sequences"):
            packed_sequences = pack_sequences(
                queries=queries,
                responses=responses,
                pack_length=args.pack_length,
                pad_token_id=tokenizer.pad_token_id,
            )
            num_new_tokens = sum(len(seq) for seq in packed_sequences.query_responses)
            
        with Timer("🔥 [Data Preparation Thread] Decoding responses", noop=True):
            decoded_responses = tokenizer.batch_decode(responses, skip_special_tokens=True)
            stop_rate = sum(int(finish_reason == "stop") for finish_reason in finish_reasons) / len(finish_reasons)
            
        with Timer("💰 [Data Preparation Thread] Calculating rewards"):
            scores, reward_metrics = reward_fn(decoded_responses, items, reasoning_gym_dataset)
            
        with Timer("🎆 [Data Preparation Thread] Calculating advantages"):
            # Calculate advantages
            scores = np.array(scores)
            print(f"[Data Preparation Thread] {len(scores)=}")
            scores_per_prompt = scores.reshape(-1, args.num_samples_per_prompt_rollout)
            global_mean_grouped_rewards = scores_per_prompt.mean(axis=-1)
            global_mean_grouped_rewards = np.repeat(
                global_mean_grouped_rewards, args.num_samples_per_prompt_rollout, axis=0
            )
            global_std_grouped_rewards = scores_per_prompt.std(axis=-1)
            global_std_grouped_rewards = np.repeat(
                global_std_grouped_rewards, args.num_samples_per_prompt_rollout, axis=0
            )
            global_advantages = (scores - global_mean_grouped_rewards) / (global_std_grouped_rewards + 1e-8)

            # Vectorized advantage calculation: create a lookup array where each index corresponds to a response mask value
            # and each value is the corresponding advantage score: index 0 is set to 0 since response masks start from 1 (1-indexed)
            lookup_advantages = np.zeros(len(global_advantages) + 1, dtype=np.float32)
            lookup_advantages[1:] = global_advantages
            packed_advantages = [
                torch.tensor(lookup_advantages[packed_mask], dtype=torch.float32)
                for packed_mask in packed_sequences.response_masks
            ]
            packed_sequences.advantages = packed_advantages
        
        with Timer("🔄 [Data Preparation Thread] Prepare collated data for each worker"):
            B = (
                len(packed_sequences.query_responses) // args.world_size
            )  # essentially doing `drop_last=True`, which is fine.
            collated_data = []
            for i in range(args.world_size):
                per_device_packed_query_responses = packed_sequences.query_responses[B * i : B * (i + 1)]
                per_device_packed_attention_masks = packed_sequences.attention_masks[B * i : B * (i + 1)]
                per_device_packed_position_ids = packed_sequences.position_ids[B * i : B * (i + 1)]
                per_device_packed_advantages = packed_sequences.advantages[B * i : B * (i + 1)]
                per_device_packed_response_masks = packed_sequences.response_masks[B * i : B * (i + 1)]

                # Shuffle the batch and collate the data
                b_inds = np.random.permutation(len(per_device_packed_query_responses))
                collated_query_responses = []
                collated_attention_masks = []
                collated_position_ids = []
                collated_response_masks = []
                collated_advantages = []
                for j in range(0, len(per_device_packed_query_responses), args.per_device_train_batch_size):
                    micro_range = b_inds[j : j + args.per_device_train_batch_size]
                    collated_query_responses.append(
                        collate_fn(
                            [per_device_packed_query_responses[idx] for idx in micro_range], tokenizer.pad_token_id
                        )
                    )
                    collated_attention_masks.append(
                        collate_fn([per_device_packed_attention_masks[idx] for idx in micro_range], 0)
                    )
                    collated_position_ids.append(
                        collate_fn([per_device_packed_position_ids[idx] for idx in micro_range], 0)
                    )
                    collated_response_masks.append(
                        collate_fn([per_device_packed_response_masks[idx] for idx in micro_range], 0)
                    )
                    collated_advantages.append(
                        collate_fn([per_device_packed_advantages[idx] for idx in micro_range], 0)
                    )
                collated_data.append(
                    {
                        "collated_query_responses": collated_query_responses,
                        "collated_attention_masks": collated_attention_masks,
                        "collated_position_ids": collated_position_ids,
                        "collated_advantages": collated_advantages,
                        "collated_response_masks": collated_response_masks,
                    }
                )
        sequence_lengths = np.array([len(response) for response in responses])
        metrics = {
            "scores": np.array(scores).mean(),
            "val/sequence_lengths": sequence_lengths.mean(),
            "val/sequence_lengths_min": sequence_lengths.min(),
            "val/sequence_lengths_max": sequence_lengths.max(),
            "val/stop_rate": stop_rate,
            **reward_metrics,
        }

        if args.save_traces:
            traces = {
                "scores": scores.tolist(),
                "finish_reasons": finish_reasons,
                "responses": responses,
                "queries": queries,
                "ground_truths": items['answer'],
                "datasets": reasoning_gym_dataset.dataset_name,
                "training_step": training_step,
                **reward_metrics,
            }
            os.makedirs(args.output_dir, exist_ok=True)
            with open(f"{args.output_dir}/traces_{args.run_name}.jsonl", "a") as f:
                json.dump(traces, f)
                f.write("\n")
        packed_sequences_Q.put(
            {
                "packed_sequences": packed_sequences,  # for debugging purposes
                "collated_data": collated_data,
                "metrics": metrics,
                "responses_count": len(responses),
                "num_new_tokens": num_new_tokens,
                "B": B,
            }
        )
    
    
        
def reward_fn(responses, items, reasoning_gym_dataset):
    metrics = {}
    formatted_responses = [extract_answer(response) for response in responses]
    correctness_rewards = [reasoning_gym_dataset.data.score_answer(formatted_response, entry) for formatted_response, entry in zip(formatted_responses, items)]
    format_rewards = soft_format_reward_func(responses)
    rewards = correctness_rewards + format_rewards
    metrics['correctness_rewards'] = correctness_rewards
    metrics['format_rewards'] = format_rewards
    metrics['rewards'] = rewards
    return rewards, metrics



def main(args: Args, model_config: ModelConfig, reward_fn: Callable):
    calculate_runtime_args(args)

    # Setup experiment tracking and seeds
    all_configs = {}
    beaker_config = None
    if is_beaker_job():
        beaker_config = maybe_get_beaker_config()
        all_configs.update(vars(beaker_config))
    all_configs.update(**asdict(args), **asdict(model_config))
    if args.with_tracking:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=all_configs,
            name=args.run_name,
            save_code=True,
            tags=[args.exp_name] + get_wandb_tags(),
        )

    # Setup tokenizer and get datasets
    tokenizer_revision = (
        model_config.model_revision if model_config.tokenizer_revision is None else model_config.tokenizer_revision
    )
    tokenizer_name = (
        model_config.tokenizer_name if model_config.tokenizer_name is not None else model_config.model_name_or_path
    )
    if tokenizer_revision != model_config.model_revision:
        # Warn user if tokenizer and model use different revisions; this is an unusual
        # use case.
        warning = f"""Requested tokenizer revision `{tokenizer_revision}` is different
                   from the model revision `{model_config.model_revision}`."""
        print(warning)
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, revision=tokenizer_revision)
    reasoning_gym_dataset = ReasoningGymDataset(args.dataset_name, args.seed, args.size, tokenizer, 'system', SYSTEM_PROMPTS['DeepSeekZero'])
    eval_dataset = ReasoningGymDataset(args.dataset_name, args.eval_seed, args.eval_size, tokenizer, 'system', SYSTEM_PROMPTS['DeepSeekZero'])
    
    bundles = [{"GPU": actor_num_gpus, "CPU": actor_num_gpus * 10} for actor_num_gpus in args.num_learners_per_node]
    pg = placement_group(bundles, strategy="PACK")
    ray.get(pg.ready())
    inits = []
    policy_group = ModelGroup(
        pg,
        PolicyTrainerRayProcess,
        args.num_learners_per_node,
        args.single_gpu_mode,
    )
    
    wandb_url = wandb.run.get_url() if args.with_tracking else None
    inits.extend(
        model.from_pretrained.remote(args, model_config, beaker_config, wandb_url, tokenizer)
        for model in policy_group.models
    )
    max_len = args.max_prompt_token_length + args.response_length
    vllm_engines = create_vllm_engines(
        args.vllm_num_engines,
        args.vllm_tensor_parallel_size,
        args.vllm_enforce_eager,
        model_config.model_name_or_path,
        model_config.model_revision,
        args.seed,
        args.vllm_enable_prefix_caching,
        max_len,
        args.vllm_gpu_memory_utilization,
        args.single_gpu_mode,
        pg=pg if args.single_gpu_mode else None,
    )
    ray.get(inits)
    print("======== ✅ all models and vLLM engines initialized =========")

    ray.get([m.setup_model_update_group.remote(vllm_engines=vllm_engines) for m in policy_group.models])
    print("======== ✅ model update group setup successfully =========")

    # Setup training
    generation_config = SamplingParams(
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.response_length,
        include_stop_str_in_output=True,
        n=args.num_samples_per_prompt_rollout,
        stop=args.stop_strings,
    )
    eval_generation_config = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.response_length,
        include_stop_str_in_output=True,
        n=1,  # since we are doing greedy sampling, don't need to generate more
        stop=args.stop_strings,
    )
    
    iter_dataloader = DataLoader(reasoning_gym_dataset, batch_size=args.num_unique_prompts_rollout, collate_fn=list_preserving_collate)
    inference_results_Q = Queue(maxsize=1)
    param_prompt_Q = Queue(maxsize=1)
    evaluation_inference_results_Q = Queue(maxsize=1)
    packed_sequences_Q = Queue(maxsize=1)
    queries_prompt_Q = Queue(maxsize=1)
    num_eval_samples = 50
    
    eval_prompt_token_ids = None
    eval_ground_truths = None
    if eval_dataset is not None:
        model_eval_input, eval_items = next(iter(eval_dataset))
        eval_prompt_token_ids = model_eval_input
    resume_training_step = 1
    thread = threading.Thread(
        target=vllm_generate_thread,
        args=(
            vllm_engines,
            generation_config,
            eval_generation_config,
            inference_results_Q,
            param_prompt_Q,
            args.num_training_steps,
            eval_prompt_token_ids,
            evaluation_inference_results_Q,
            args.eval_freq,
            resume_training_step,
        ),
    )
    
    thread.start()
    print("======== ✅ vllm generate thread starts =========")
    
    packing_thread = threading.Thread(
        target=data_preparation_thread,
        args=(
            reward_fn,
            inference_results_Q,
            packed_sequences_Q,
            queries_prompt_Q,
            args,
            tokenizer,
            args.num_training_steps,
            reasoning_gym_dataset,
        ),
    )
    
    packing_thread.start()
    print("======== ✅ data preparation thread starts =========")
    model_inputs, items = next(iter(iter_dataloader))
    queries_prompt_Q.put((model_inputs, items))
    param_prompt_Q.put((None, model_inputs))
    
    episode = 0
    num_total_tokens = 0
    start_time = time.time()
    for training_step in range(resume_training_step, args.num_training_steps + 1):
        episode += (args.num_unique_prompts_rollout * args.num_samples_per_prompt_rollout)
        
        if training_step != 1:
            model_inputs, items = next(iter(iter_dataloader))
            with Timer("🔄 Loading weights using shared memory"):
                ray.get([m.broadcast_to_vllm.remote() for m in policy_group.models])
            
            queries_prompt_Q.put((model_inputs, items))
            param_prompt_Q.put((None, model_inputs))
            
        with Timer("[Main Thread] 📦 Getting packed sequences from thread"):
            packed_data = packed_sequences_Q.get()
            data_thread_metrics = packed_data["metrics"]
            B = packed_data["B"]
            collated_data = packed_data["collated_data"]
            num_total_tokens += packed_data["num_new_tokens"]
        
            if B == 0:
                print("[Main Thread] 🤡 After packing, there is not enough data to train")
                continue

            # ------------------------------------------------------------------------------------------------
            # Train the model
            with Timer("[Main Thread] 🗡️ Training"):
                metrics_list: List[dict[str, float]] = ray.get(
                    [
                        policy_group.models[i].train.remote(
                            **collated_data[i],
                            pad_token_id=tokenizer.pad_token_id,
                            num_mini_batches=args.num_mini_batches,
                        )
                        for i in range(args.world_size)
                    ]
                )
                average_metrics = {k: sum(m[k] for m in metrics_list) / len(metrics_list) for k in metrics_list[0]}
                metrics = {
                    "episode": episode,
                    "training_step": training_step,
                    "val/num_total_tokens": num_total_tokens,
                    "epoch": episode / len(reasoning_gym_dataset),
                    "tokens_per_second": num_total_tokens / (time.time() - start_time),
                    **data_thread_metrics,
                    **average_metrics,
                }
                    
                if args.save_freq > 0 and training_step % args.save_freq == 0:
                    with Timer("[Main Thread] 🗡️ Saving model"):
                        checkpoint_dir = f"{args.output_dir}_checkpoints"
                        step_dir = os.path.join(checkpoint_dir, f"step_{training_step}")
                        print(f"Saving model at step {training_step} to {step_dir}")
                        ray.get([policy_group.models[i].save_model.remote(step_dir) for i in range(args.world_size)])
    
    
    print(f"Saving final model at step {training_step} to {args.output_dir}")
    with Timer("[Main Thread] 🗡️ Saving model"):
            ray.get([policy_group.models[i].save_model.remote(args.output_dir) for i in range(args.world_size)])

    thread.join()
    print("======== ✅ vllm generate thread ends =========")
    packing_thread.join()
    print("======== ✅ data preparation thread ends =========")
    ray.shutdown()
    
    if (
        args.try_auto_save_to_beaker
        and is_beaker_job()
        and len(beaker_config.beaker_dataset_id_urls) > 0
        and args.output_dir.rstrip("/") != "/output"
    ):
        shutil.copytree(args.output_dir, "/output", dirs_exist_ok=True)
    print("finished training")
    
    accelerator = Namespace()
    accelerator.is_main_process = True  # hack
    if args.push_to_hub:
        print("Pushing model to hub")
        push_folder_to_hub(
            accelerator,
            args.output_dir,
            args.hf_repo_id,
            args.hf_repo_revision,
        )
                        
        
if __name__ == "__main__":
    parser = ArgumentParserPlus((GRPOArgs, ModelConfig))
    args, model_config = parser.parse_args_into_dataclasses()
    assert isinstance(args, GRPOArgs)
    assert isinstance(model_config, ModelConfig)
    main(args, model_config, reward_fn)
    