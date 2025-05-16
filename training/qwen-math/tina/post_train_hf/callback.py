import copy
import logging
import shutil

import numpy as np
import pandas as pd
import torch
import wandb
from tina.post_train_hf.hub import push_to_hub_revision
from tina.utils.prompt import FIXED_PROMPT_FOR_EVALUATION, OPEN_R1_SYSTEM_PROMPT
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class FixedPromptEvaluationCallback(TrainerCallback):
    def __init__(
        self,
        system_prompt=OPEN_R1_SYSTEM_PROMPT,
        prompt=FIXED_PROMPT_FOR_EVALUATION,
        max_generation_length=4096,
        eval_steps=100,
    ):

        self.system_prompt = system_prompt
        self.prompt = prompt
        self.max_generation_length = max_generation_length
        self.eval_steps = eval_steps
        self.completion_table = {
            "step": [],
            "prompt": [],
            "completion": [],
        }

    def on_init_end(self, args, state, control, processing_class=None, **kwargs):
        tokenizer = processing_class
        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": self.prompt}]
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        self.tokenized_prompt = tokenizer(input_text, return_tensors="pt")

    def on_step_end(self, args, state, control, model=None, processing_class=None, **kwargs):
        if state.global_step % self.eval_steps == 0:
            if state.is_world_process_zero:
                completion = self.eval_prompt(model, processing_class)
                self.completion_table["step"].append(str(state.global_step))
                self.completion_table["prompt"].append(self.prompt)
                self.completion_table["completion"].append(completion)
                df = pd.DataFrame(self.completion_table)
                wandb.log({"completions": wandb.Table(dataframe=df)})

    def eval_prompt(self, model, tokenizer):
        if hasattr(model, "peft_config"):
            model.peft_config["default"].inference_mode = True

        self.tokenized_prompt.to(model.device)
        outputs = model.generate(
            **self.tokenized_prompt,
            max_length=self.max_generation_length,
            temperature=0.01,  # Very low temperature
            top_k=1,  # Only consider the most likely token
            top_p=1.0,  # Disable nucleus sampling or set to a high value
        )
        completion = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if hasattr(model, "peft_config"):
            model.peft_config["default"].inference_mode = False

        return completion


class GradientClippingLoggerCallback(TrainerCallback):
    def on_step_end(self, args, state, control, model=None, processing_class=None, **kwargs):
        self.clipped_grad_norm = np.sqrt(
            sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None)
        )
        if state.is_world_process_zero:
            wandb.log({"clipped_grad_norm": self.clipped_grad_norm})

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            logs["clipped_grad_norm"] = self.clipped_grad_norm


class DummyConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class PushToHubRevisionCallback(TrainerCallback):
    def __init__(self, dataset_name, use_peft):
        self.dataset_name = dataset_name
        self.use_peft = use_peft

        self.pending_futures = []  # Track pending push operations

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            global_step = state.global_step

            # Create merged model directory
            if self.use_peft:
                ckpt_model_dir = f"{args.output_dir}/checkpoint-{global_step}-merged"
                original_model = kwargs["model"]  # Don't pop it, keep it intact
                model_to_save = copy.deepcopy(original_model).merge_and_unload()
                model_to_save.save_pretrained(ckpt_model_dir)
            else:
                # this dir is already created by the HF Trainer, no need to manually save
                ckpt_model_dir = f"{args.output_dir}/checkpoint-{global_step}"

            tokenizer = kwargs.get("tokenizer") or kwargs.get("processing_class")
            if tokenizer is None:
                raise ValueError("Tokenizer or processing_class must be provided.")
            tokenizer.save_pretrained(ckpt_model_dir)

            dummy_config = DummyConfig(
                hub_model_id=args.hub_model_id,
                hub_model_revision=self.dataset_name,
                checkpoint=f"checkpoint-{global_step}",
                output_dir=ckpt_model_dir,
                dataset_name=self.dataset_name,
            )

            # Start the push operation
            future = push_to_hub_revision(dummy_config, extra_ignore_patterns=["*.pt"])

            # Store the future and directory path for cleanup later
            self.pending_futures.append((future, ckpt_model_dir))

            # Check and clean up any completed pushes
            if self.use_peft:
                self._cleanup_completed_pushes()

        return control

    def _cleanup_completed_pushes(self):
        """Check pending futures and remove directories for completed pushes."""
        still_pending = []
        for future, dir_path in self.pending_futures:
            if future.done():
                if self.use_peft:
                    # The push is complete, safe to delete the directory
                    try:
                        shutil.rmtree(dir_path)
                        logger.info(f"\nCleaned up merged model directory: {dir_path}\n")
                    except Exception as e:
                        logger.error(f"\nFailed to clean up directory {dir_path}: {e}\n")
            else:
                # Push is still in progress, keep in pending list
                still_pending.append((future, dir_path))

        self.pending_futures = still_pending

    def on_train_end(self, args, state, control, **kwargs):
        """Make sure to clean up any remaining directories at the end of training."""
        if state.is_world_process_zero and self.use_peft:
            # Wait for all pending pushes to complete
            logger.info(f"\nCleaned up for lora models.")
            for future, dir_path in self.pending_futures:
                future.result()  # Wait for completion
                try:
                    shutil.rmtree(dir_path)
                    logger.info(f"\nCleaned up merged model directory: {dir_path}\n")
                except Exception as e:
                    logger.error(f"\nFailed to clean up directory {dir_path}: {e}\n")

            self.pending_futures = []
