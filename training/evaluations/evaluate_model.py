#!/usr/bin/env python

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml
from tqdm import tqdm
from vllm import LLM, SamplingParams

import reasoning_gym
from reasoning_gym.utils import SYSTEM_PROMPTS, extract_answer


@dataclass
class DatasetConfig:
    dataset: str
    size: Optional[int] = None
    seed: Optional[int] = None
    params: Dict[str, Any] = None

    def __post_init__(self):
        if self.params is None:
            self.params = {}


@dataclass
class CategoryConfig:
    category: str
    datasets: List[DatasetConfig]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CategoryConfig":
        datasets = [DatasetConfig(**d) for d in data["datasets"]]
        return cls(category=data["category"], datasets=datasets)


@dataclass
class EvalConfig:
    model_path: str
    max_tokens: int
    temperature: float
    top_p: float
    output_dir: str
    save_metadata: bool
    save_full_results: bool
    categories: List[CategoryConfig]

    # Optional: you can provide a system prompt name (looked up in SYSTEM_PROMPTS)
    developer_prompt: Optional[str] = None
    developer_role: str = "system"

    # NEW FIELD: How many times each question is evaluated
    eval_repeats: int = 1

    @classmethod
    def from_yaml(cls, path: str) -> "EvalConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        categories = [CategoryConfig.from_dict(cat) for cat in data["categories"]]
        data["categories"] = categories
        return cls(**data)


class LocalModelEvaluator:
    def __init__(
        self,
        model_path: str,
        config: EvalConfig,
        device: str = "cuda:0",
        batch_size: int = 1,
        verbose: bool = False,
    ):
        self.config = config
        self.device = device
        self.batch_size = batch_size
        self.verbose = verbose

        # Load model and tokenizer
        self.llm = LLM(model=model_path)
        self.tokenizer = self.llm.get_tokenizer()
        self.sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
        )

        self.start_time = datetime.now()
        # If you have a system prompt, retrieve it from SYSTEM_PROMPTS
        self.developer_prompt = None
        if self.config.developer_prompt:
            self.developer_prompt = SYSTEM_PROMPTS[self.config.developer_prompt]
        self.developer_role = self.config.developer_role

    def get_model_response(self, question: str) -> str:
        """
        Generates a single response to the given question and returns the
        raw text of that response.
        """
        # Build a "chat" prompt if developer_prompt is available
        chat = []
        if self.developer_prompt:
            chat.append({"role": self.developer_role, "content": self.developer_prompt})
        chat.append({"role": "user", "content": question})

        # Some Hugging Face chat-friendly models use a convenience method like below:
        prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        response = self.llm.generate(prompt, self.sampling_params, use_tqdm=False)
        # Extract the text from the response
        response = response[0].outputs[0].text

        if self.verbose:
            print(f"[Prompt]\n{question}\n[Response]\n{response}\n{'-'*60}")

        return response

    def process_entry(self, dataset, entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate one question from the dataset `eval_repeats` times, then
        average the score. We also keep track of the best (max) score
        and store each completion for potential debugging.
        """
        all_completions = []
        for _ in range(self.config.eval_repeats):
            try:
                raw_response = self.get_model_response(entry["question"])
                model_answer = extract_answer(raw_response)
                score = dataset.score_answer(answer=model_answer, entry=entry)
                score = 0.0 if score < 1 else score
                all_completions.append(
                    {
                        "model_answer": model_answer,
                        "full_model_response": raw_response,
                        "score": score,
                    }
                )
            except Exception as e:
                # If there's an error on a single repetition, store it and continue
                all_completions.append(
                    {
                        "model_answer": None,
                        "full_model_response": "",
                        "score": 0.0,
                        "error": str(e),
                    }
                )

        # Compute statistics across all runs
        scores = [c["score"] for c in all_completions]
        mean_score = sum(scores) / len(scores)
        best_score = max(scores)

        return {
            "question": entry["question"],
            "expected_answer": str(entry["answer"]),
            "best_score": best_score,
            "mean_score": mean_score,
            "completions": all_completions,
        }

    def evaluate_dataset(self, category_name: str, dataset_config: DatasetConfig) -> Dict[str, Any]:
        """
        Loads the dataset, processes each entry, and then computes
        the overall average across all entries.
        """
        dataset_name = dataset_config.dataset
        params = {
            **dataset_config.params,
            "size": dataset_config.size,
            "seed": dataset_config.seed,
        }
        dataset = reasoning_gym.create_dataset(dataset_name, **params)
        entries = list(dataset)

        results = []
        for entry in tqdm(entries, desc=f"Processing {dataset_name}"):
            results.append(self.process_entry(dataset, entry))

        # Summarize the entire dataset
        total_mean_score = sum(r["mean_score"] for r in results)
        avg_score = total_mean_score / len(results) if results else 0.0

        return {
            "name": dataset_name,
            "category": category_name,
            "average_score": avg_score,
            "total_examples": len(results),
            "config": params,
            "results": results,
        }

    def evaluate_all(self) -> Dict[str, Any]:
        """
        Runs evaluation on all categories/datasets.
        """
        cat_results = []
        for cat in self.config.categories:
            datasets = []
            for ds_cfg in cat.datasets:
                datasets.append(self.evaluate_dataset(cat.category, ds_cfg))
            cat_results.append({"name": cat.category, "datasets": datasets})

        return {
            "metadata": {
                "timestamp": self.start_time.isoformat(),
                "model": self.config.model_path,
                "device": self.device,
                "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "eval_repeats": self.config.eval_repeats,
            },
            "categories": cat_results,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--category")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # Load config from YAML
    config = EvalConfig.from_yaml(args.config)

    # Command-line overrides
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.category:
        # Filter categories if specified
        config.categories = [c for c in config.categories if c.category == args.category]
        if not config.categories:
            print(f"Category '{args.category}' not found.")
            return 1

    evaluator = LocalModelEvaluator(
        model_path=config.model_path,
        config=config,
        device=args.device,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )

    results = evaluator.evaluate_all()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(config.output_dir) / f"local_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {out_dir / 'results.json'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
