import argparse
from typing import Any

from transformers import AutoModel, AutoTokenizer

import reasoning_gym
from reasoning_gym.dataset import ProceduralDataset
from reasoning_gym.utils import SYSTEM_PROMPTS, extract_answer

from ..utils import ReasoningGymDataset


def get_model_response(model, prompt: str) -> str:
    # TODO
    return model.generate(prompt)


def process_entry(model, dataset: ProceduralDataset, entry: dict[str, Any]) -> float:
    model_response = get_model_response(model, entry["question"])
    model_answer = extract_answer(model_response)
    score = dataset.score_answer(answer=model_answer, entry=entry)
    return score


def evaluate(model, dataset: ReasoningGymDataset) -> float:
    procedural_dataset = dataset.data

    total_score, n = 0.0, 0
    for entry in procedural_dataset:
        score = process_entry(model, procedural_dataset, entry)
        total_score += score
        n += 1

    return total_score / n


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)

    procedural_dataset = reasoning_gym.create_dataset(args.dataset_name, seed=args.dataset_seed, size=args.dataset_size)

    dataset = ReasoningGymDataset(
        tokenizer=tokenizer,
        procedural_dataset=procedural_dataset,
        developer_prompt=SYSTEM_PROMPTS[args.developer_prompt],
    )

    score = evaluate(model, dataset)
    print(f"Score: {score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--dataset-size", type=int, default=10000)
    parser.add_argument("--dataset-seed", type=int, default=2)
    parser.add_argument("--developer-prompt", type=str, default="DeepSeekZero")
    args = parser.parse_args()
