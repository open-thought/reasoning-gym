import argparse
import os
from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--adapter_type", type=str, default="grpo_curated_open_r1")
    parser.add_argument("--ckpt", type=str, default="checkpoint-500")
    return parser.parse_args()


if __name__ == "__main__":
    args = argparser()

    ckpt_dir = os.environ["CKPT_DIR"]
    ckpt = args.ckpt
    adapter_type = args.adapter_type
    model_name = args.model_name

    base_model_name_or_path = f"{ckpt_dir}/models/{model_name}/base"
    adapter_model_name_or_path = f"{ckpt_dir}/models/{model_name}/{adapter_type}/{ckpt}"
    merged_model_name_or_path = f"{ckpt_dir}/models/{model_name}/{adapter_type}/{ckpt}-merged"

    print("Merged model will be saved to: ", merged_model_name_or_path)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto") # Automatically distributes across available GPUs

    model = PeftModel.from_pretrained(base_model, adapter_model_name_or_path)
    model = model.merge_and_unload()

    model.save_pretrained(merged_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
    tokenizer.save_pretrained(merged_model_name_or_path)
