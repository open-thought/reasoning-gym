import logging
import os
import sys
from datetime import datetime

import datasets
import torch
import transformers
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from tina.config import ModelPTConfig
from tina.post_train_hf.callback import (
    FixedPromptEvaluationCallback,
    GradientClippingLoggerCallback,
    PushToHubRevisionCallback,
)
from tina.post_train_hf.grpo_config import GRPOConfig  # use this new one for Dr.GRPO
from tina.post_train_hf.grpo_trainer import GRPOTrainer  # use this new one for Dr.GRPO
from tina.post_train_hf.preprocess import make_conv_for_grpo
from tina.post_train_hf.rewards import (
    accuracy_reward,
    format_reward,
    get_cosine_scaled_reward,
    get_repetition_penalty_reward,
    len_reward,
    reasoning_steps_reward,
    tag_count_reward,
)
from tina.utils.chat_template import DEFAULT_CHAT_TEMPLATE, REASON_CHAT_TEMPLATE
from tina.utils.constant import RL_POST_TRAIN_DATASET_MAP
from tina.utils.prompt import OPEN_R1_SYSTEM_PROMPT, OPEN_RS_SYSTEM_PROMPT
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import ModelConfig, TrlParser  # GRPOTrainer, GRPOConfig


def main():
    parser = TrlParser((ModelPTConfig, GRPOConfig, ModelConfig))
    pt_args, training_args, model_args = parser.parse_args_and_config()
    set_seed(training_args.seed)

    os.environ["WANDB_PROJECT"] = "tina_model_post_training"

    ################
    # Set up logging
    ################

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Post training parameters {pt_args}")
    logger.info(f"Training parameters {training_args}")

    #####################
    # Set up output paths
    #####################

    current_time = datetime.now()
    formatted_datetime = current_time.strftime("%Y_%m_%d_%H_%M_%S")

    model_name_or_path = model_args.model_name_or_path
    ckpt_dir = os.environ["CKPT_DIR"]
    ckpt_prefix = f"{ckpt_dir}/models/{model_name_or_path}"
    if model_args.use_peft:
        ckpt_postfix = f"{pt_args.model_post_train_type}_{pt_args.model_post_train_dataset_name}"
    else:
        ckpt_postfix = f"full_{pt_args.model_post_train_type}_{pt_args.model_post_train_dataset_name}"

    model_args.model_name_or_path = f"{ckpt_prefix}/base"
    training_args.output_dir = f"{ckpt_prefix}/{ckpt_postfix}"
    # training_args.hub_model_id = f"{training_args.hub_model_id}_{ckpt_postfix}"
    training_args.run_name = f"{model_name_or_path}_{ckpt_postfix}_{formatted_datetime}"

    training_args.hub_model_id = f"{training_args.hub_model_id}/{model_name_or_path}"

    #######################################################################
    # Load and preprocess dataset (tokenization is handled by GRPO Trainer)
    #######################################################################

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if "Llama" in model_args.model_name_or_path:
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
    elif "Qwen" in model_args.model_name_or_path:
        tokenizer.pad_token = "<|fim_pad|>"
    tokenizer.chat_template = REASON_CHAT_TEMPLATE

    model_post_train_dataset_name = RL_POST_TRAIN_DATASET_MAP[pt_args.model_post_train_dataset_name]
    if pt_args.model_post_train_dataset_config is not None:
        train_dataset = load_dataset(
            model_post_train_dataset_name, split="train", name=pt_args.model_post_train_dataset_config
        )
    else:
        train_dataset = load_dataset(model_post_train_dataset_name, split="train")
    # required by GRPOTrainer: (prompt, solution) columns
    if "solution" not in train_dataset.column_names and "answer" in train_dataset.column_names:
        train_dataset = train_dataset.rename_column("answer", "solution")

        # Wrap the 'solution' values in $...$
        def wrap_in_math(example):
            return {"solution": f"${example['solution']}$"}

        # Apply the transformation to the entire dataset
        train_dataset = train_dataset.map(wrap_in_math)
    if "problem" not in train_dataset.column_names and "question" in train_dataset.column_names:
        train_dataset = train_dataset.rename_column("question", "problem")
    if "problem" not in train_dataset.column_names and "prompt" in train_dataset.column_names:
        train_dataset = train_dataset.rename_column("prompt", "problem")
    if "messages" in train_dataset.column_names:
        train_dataset = train_dataset.remove_columns("messages")

    # handle deepscaler separately
    if "deepscaler" in pt_args.model_post_train_dataset_name:
        train_dataset = train_dataset.rename_column("solution", "solution_archive")
        train_dataset = train_dataset.rename_column("answer", "solution")

        # Wrap the 'solution' values in $...$
        def wrap_in_math(example):
            return {"solution": f"${example['solution']}$"}

        # Apply the transformation to the entire dataset
        train_dataset = train_dataset.map(wrap_in_math)

    SYSTEM_PROMPT = OPEN_RS_SYSTEM_PROMPT if "open-rs" in model_post_train_dataset_name else OPEN_R1_SYSTEM_PROMPT
    train_dataset = train_dataset.map(make_conv_for_grpo, fn_kwargs={"system_prompt": SYSTEM_PROMPT})

    ######################
    # Initialize the model
    ######################

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=model_args.attn_implementation,
        use_cache=False if training_args.gradient_checkpointing else True,
    )

    if model_args.use_peft:
        logger.info(
            f"\n Using PEFT with {model_args.lora_r} rank, {model_args.lora_alpha} alpha, {model_args.lora_dropout} dropout."
        )
        peft_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=model_args.lora_target_modules,
            inference_mode=False,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, peft_config)

    #############################
    # Initialize the GRPO trainer
    #############################

    RL_POST_TRAIN_REWARD_MAP = {
        "accuracy": accuracy_reward,
        "format": format_reward,
        "tag_count": tag_count_reward,
        "length": len_reward,
        "reasoning_steps": reasoning_steps_reward,
        "cosine": get_cosine_scaled_reward(
            min_value_wrong=pt_args.cosine_min_value_wrong,
            max_value_wrong=pt_args.cosine_max_value_wrong,
            min_value_correct=pt_args.cosine_min_value_correct,
            max_value_correct=pt_args.cosine_max_value_correct,
            max_len=pt_args.cosine_max_len,
        ),
        "repetition_penalty": get_repetition_penalty_reward(
            ngram_size=pt_args.repetition_n_grams,
            max_penalty=pt_args.repetition_max_penalty,
        ),
    }
    rl_reward_funcs = [RL_POST_TRAIN_REWARD_MAP[func] for func in pt_args.rl_post_train_reward_funcs]
    training_args.reward_weights = pt_args.rl_post_train_reward_weights

    if model_args.use_peft:
        callbacks = [
            FixedPromptEvaluationCallback(system_prompt=OPEN_R1_SYSTEM_PROMPT, eval_steps=training_args.save_steps),
            # PushToHubRevisionCallback(dataset_name=pt_args.model_post_train_dataset_name, use_peft=model_args.use_peft)
        ]
    else:
        callbacks = [
            FixedPromptEvaluationCallback(system_prompt=OPEN_R1_SYSTEM_PROMPT, eval_steps=training_args.save_steps),
            GradientClippingLoggerCallback(),
            # PushToHubRevisionCallback(dataset_name=pt_args.model_post_train_dataset_name, use_peft=model_args.use_peft)
        ]

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=rl_reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        callbacks=callbacks,
    )

    #########################
    # Training and Evaluation
    #########################

    logger.info(f"\nStarting training for {training_args.num_train_epochs} epochs.")

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

    train_result = trainer.train(resume_from_checkpoint=ckpt)
    train_metrics = train_result.metrics
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    trainer.save_state()
    trainer.push_to_hub(
        commit_message=f"Add checkpoint {training_args.max_steps} post-trained on {pt_args.model_post_train_dataset_name}"
    )

    del trainer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
