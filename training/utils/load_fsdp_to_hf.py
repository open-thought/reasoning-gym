#!/usr/bin/env python
# encoding: utf-8
from collections import defaultdict
from glob import glob

import fire
import torch
from huggingface_hub import HfApi, create_repo
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def main(fsdp_checkpoint_path, huggingface_model_path, output_path, push_to_hub=True, hub_token="", private=True):
    state_dict = defaultdict(list)

    world_size = 4
    for rank in range(world_size):
        filepath = f"{fsdp_checkpoint_path}/model_world_size_{world_size}_rank_{rank}.pt"
        print("loading", filepath)
        this_state_dict = torch.load(filepath)
        for key, value in this_state_dict.items():
            state_dict[key].append(value.to_local())

    for key in state_dict:
        state_dict[key] = torch.cat(state_dict[key], dim=0)

    config = AutoConfig.from_pretrained(huggingface_model_path)
    model = AutoModelForCausalLM.from_config(config)
    model.load_state_dict(state_dict)

    model.save_pretrained(output_path, max_shard_size="10GB")

    tokenizer = AutoTokenizer.from_pretrained(huggingface_model_path)
    tokenizer.save_pretrained(output_path)

    # Push to hub if requested
    if push_to_hub:
        if not output_path:
            raise ValueError("output path must be provided when push_to_hub=True")

        print(f"Pushing model to Hugging Face Hub: {output_path}")

        # Create repository if it doesn't exist
        api = HfApi(token=hub_token)
        try:
            create_repo(repo_id=output_path, private=private, exist_ok=True, token=hub_token)
            print(f"Repository {output_path} created or already exists")
        except Exception as e:
            print(f"Repository creation info: {e}")

        # Push model and tokenizer to hub
        model.push_to_hub(output_path, token=hub_token, private=private)
        tokenizer.push_to_hub(output_path, token=hub_token, private=private)

        print(f"✅ Model successfully pushed to https://huggingface.co/{output_path}")


if __name__ == "__main__":
    fire.Fire(main)
