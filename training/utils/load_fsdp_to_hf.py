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


if __name__ == "__main__":
    fire.Fire(main)
