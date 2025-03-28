#!/usr/bin/env python
"""
Local evaluation script for reasoning gym datasets.

This script evaluates local LLM performance on reasoning gym datasets.

Usage:
    python local_eval.py --model-path PATH --config config.yaml [options]

Options:
    --model-path PATH         Path to local model directory/file
    --config CONFIG          Path to configuration file
    --output-dir DIR         Override output directory specified in config
    --category CATEGORY      Evaluate only datasets from this category
    --device DEVICE         Device to run model on (cuda:0, cpu, etc.)
    --batch-size N          Batch size for inference
    --verbose               Print detailed model responses
    --debug                 Enable debug logging
"""

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml
from eval_config import CategoryConfig, DatasetConfig, EvalConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import reasoning_gym
from reasoning_gym.utils import SYSTEM_PROMPTS, extract_answer

try:
    import torch.distributed as dist
    import torch.distributed._tensor as dt

    HAS_DIST_TENSOR = True
except ImportError:
    HAS_DIST_TENSOR = False

import importlib
from collections import OrderedDict


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
        datasets = [DatasetConfig(**dataset) for dataset in data["datasets"]]
        return cls(category=data["category"], datasets=datasets)


@dataclass
class EvalConfig:
    # Model configuration
    model: str
    hf_path: Optional[str]
    max_tokens: int
    temperature: float
    top_p: float
    developer_prompt: str
    developer_role: str

    # Output configuration
    output_dir: str
    save_metadata: bool
    save_full_results: bool

    # Categories
    categories: List[CategoryConfig]

    @classmethod
    def from_yaml(cls, path: str) -> "EvalConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # Convert categories data to CategoryConfig objects
        categories = [CategoryConfig.from_dict(cat) for cat in data["categories"]]

        # Remove categories from data dict and pass remaining args to constructor
        data["categories"] = categories

        return cls(**data)

    def get_system_prompt(self) -> str:
        """Get the system prompt to use for evaluation."""
        return SYSTEM_PROMPTS[self.developer_prompt]


class LocalModelEvaluator:
    """Evaluates local models on reasoning datasets."""

    def __init__(
        self,
        model_path: str,
        config: EvalConfig,
        hf_path: Optional[str] = None,
        device: str = "cuda:0",
        batch_size: int = 1,
        verbose: bool = False,
        debug: bool = False,
    ):
        """Initialize the evaluator.

        Args:
            model_path: Path to base model or model identifier
            config: Evaluation configuration
            hf_path: Path to HuggingFace checkpoint directory containing trained weights
            device: Device to run model on
            batch_size: Batch size for inference
            verbose: Whether to print detailed responses
            debug: Whether to enable debug logging
        """
        self.config = config
        self.device = device
        self.batch_size = batch_size
        self.verbose = verbose
        self.developer_prompt = config.developer_prompt
        self.developer_role = config.developer_role

        # Set up logging
        self.logger = logging.getLogger("LocalModelEvaluator")
        if debug:
            self.logger.setLevel(logging.DEBUG)

        # Load base model and tokenizer
        self.logger.info(f"Loading base model from {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if "cuda" in device else torch.float32,
            device_map=device,
            trust_remote_code=True,  # Required for Qwen models
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)  # Required for Qwen models

        # Try to load from the huggingface folder if it exists
        if hf_path:
            self.logger.info(f"Loading trained weights from {hf_path}")

            # First load the base model to ensure compatibility
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16 if "cuda" in device else torch.float32,
                device_map=device,
                trust_remote_code=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

            # Path to distributed checkpoint
            state_dict_path = Path(hf_path) / "global_step_312/actor/model_world_size_2_rank_0.pt"

            if state_dict_path.exists():
                self.logger.info(f"Loading distributed state dict from {state_dict_path}")

                # Load the checkpoint with special handling
                try:
                    # Custom loading with mapping to CPU first
                    checkpoint = torch.load(state_dict_path, map_location="cpu")

                    # Create a clean state dict
                    clean_state_dict = OrderedDict()

                    # Process each key-value pair
                    for k, v in checkpoint.items():
                        # Remove the "module." prefix if present
                        if k.startswith("module."):
                            k = k[7:]

                        # Handle DTensor objects
                        if hasattr(v, "_local_tensor"):
                            # For distributed tensor - extract local tensor
                            v = v._local_tensor
                        elif str(type(v)).find("DTensor") > 0:
                            # Alternative way to detect DTensor
                            v = v.to_local() if hasattr(v, "to_local") else v

                        # Add to clean state dict
                        clean_state_dict[k] = v

                    # Load the cleaned state dict
                    missing, unexpected = self.model.load_state_dict(clean_state_dict, strict=False)

                    self.logger.info(
                        f"Successfully loaded checkpoint. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}"
                    )
                    if self.verbose:
                        self.logger.info(f"Missing keys: {missing[:10]}...")
                        self.logger.info(f"Unexpected keys: {unexpected[:10]}...")

                except Exception as e:
                    self.logger.error(f"Error loading distributed checkpoint: {e}")
                    raise RuntimeError(f"Failed to load distributed checkpoint: {e}")
            else:
                raise FileNotFoundError(f"Checkpoint file not found: {state_dict_path}")

        # Move model to device if needed
        if not hasattr(self.model, "hf_device_map"):
            self.model.to(device)

        # Metadata
        self.start_time = datetime.now()

    def get_model_response(self, prompt: str) -> list[str]:
        """Get response from local model."""
        if self.verbose:
            self.logger.info(f"Prompt: {prompt}")

        # Prepare full prompt with system message
        chat = []
        if self.developer_prompt is not None:
            chat.append({"role": self.developer_role, "content": self.config.get_system_prompt()})
        chat.append({"role": "user", "content": prompt})

        # Use the model's chat template
        full_prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        # Tokenize
        inputs = self.tokenizer(
            full_prompt, return_tensors="pt", truncation=True, max_length=self.tokenizer.model_max_length
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True if self.config.temperature > 0 else False,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True).strip()

        if self.verbose:
            self.logger.info(f"Response: {response}")

        return [response]

    def process_entry(
        self,
        dataset: reasoning_gym.dataset.ProceduralDataset,
        entry: dict[str, Any],
        entry_index: int,
        dataset_name: str,
    ) -> dict[str, Any]:
        """Process a single dataset entry."""
        try:
            # Get model response
            responses = self.get_model_response(entry["question"])
            response = responses[0]

            # Extract and score answer
            model_answer = extract_answer(response)

            if self.verbose:
                self.logger.info(f"Question: {entry['question']}")
                self.logger.info(f"Expected: {entry['answer']}")
                self.logger.info(f"Model answer: {model_answer}")
                self.logger.info(f"Full response: {response}")

            # Score the answer
            score = dataset.score_answer(answer=model_answer, entry=entry)

            if self.verbose:
                self.logger.info(f"Score: {score}")
                self.logger.info("-" * 40)

            result = {
                "question": entry["question"],
                "expected_answer": str(entry["answer"]),
                "best_model_answer": model_answer,
                "best_full_model_response": response,
                "best_score": score,
                "mean_score": score,
                "completions": [{"model_answer": model_answer, "full_model_response": response, "score": score}],
            }

            return result

        except Exception as e:
            self.logger.error(f"Error processing entry {entry_index}:")
            self.logger.error(f"Question: {entry['question']}")
            self.logger.error(f"Error: {str(e)}")
            return {
                "question": entry["question"],
                "expected_answer": str(entry["answer"]),
                "best_model_answer": None,
                "best_full_model_response": None,
                "best_score": 0.0,
                "mean_score": 0.0,
                "error": str(e),
                "completions": [],
            }

    def evaluate_dataset(self, category_name: str, dataset_config: DatasetConfig) -> dict[str, Any]:
        """Evaluate a single dataset.

        Args:
            category_name: Name of the category
            dataset_config: Configuration for the dataset

        Returns:
            Dict with evaluation results
        """
        dataset_name = dataset_config.dataset
        self.logger.info(f"Evaluating dataset: {dataset_name}")

        try:
            # Create dataset
            dataset_params = {**dataset_config.params, "size": dataset_config.size, "seed": dataset_config.seed}
            dataset = reasoning_gym.create_dataset(dataset_name, **dataset_params)

            # Get all entries
            all_entries = list(dataset)

            # Process entries with progress bar
            results = []
            for idx, entry in tqdm(enumerate(all_entries), desc=f"Processing {dataset_name}"):
                result = self.process_entry(dataset, entry, idx, dataset_name)
                results.append(result)

            # Calculate metrics
            total_score = sum(r["best_score"] for r in results)
            average_score = total_score / len(results) if results else 0

            return {
                "name": dataset_name,
                "category": category_name,
                "average_best_score": average_score,
                "average_mean_score": average_score,
                "total_examples": len(results),
                "config": dataset_params,
                "system_prompt": self.config.get_system_prompt(),
                "results": results,
            }

        except Exception as e:
            self.logger.error(f"Error evaluating dataset {dataset_name}: {str(e)}")
            return {
                "name": dataset_name,
                "category": category_name,
                "average_best_score": 0.0,
                "average_mean_score": 0.0,
                "total_examples": 0,
                "config": dataset_params,
                "system_prompt": self.config.get_system_prompt(),
                "error": str(e),
                "results": [],
            }

    def evaluate_all(self) -> dict[str, Any]:
        """Evaluate all categories and datasets.

        Returns:
            Dict with evaluation results
        """
        self.logger.info(f"Starting evaluation of {len(self.config.categories)} categories")

        category_results = []
        for category in self.config.categories:
            self.logger.info(f"Evaluating category: {category.category}")

            dataset_results = []
            for dataset_config in category.datasets:
                result = self.evaluate_dataset(category.category, dataset_config)
                dataset_results.append(result)

            category_results.append({"name": category.category, "datasets": dataset_results})

        results = {
            "metadata": {
                "timestamp": self.start_time.isoformat(),
                "model": self.config.model,
                "device": self.device,
                "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
            },
            "categories": category_results,
        }

        return results


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description="Evaluate local models on reasoning datasets")
    parser.add_argument("--model-path", required=True, help="Path to base model or model identifier")
    parser.add_argument("--hf-path", help="Path to HuggingFace checkpoint directory containing trained weights")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--output-dir", help="Override output directory specified in config")
    parser.add_argument("--category", help="Evaluate only datasets from this category")
    parser.add_argument("--device", default="cuda:0", help="Device to run model on")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--verbose", action="store_true", help="Print detailed responses")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load configuration using our new config class
    config = EvalConfig.from_yaml(args.config)

    # Apply command line overrides
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.hf_path:
        config.hf_path = args.hf_path

    # Filter categories if specified
    if args.category:
        config.categories = [cat for cat in config.categories if cat.category == args.category]
        if not config.categories:
            print(f"Error: Category '{args.category}' not found in configuration")
            return 1

    try:
        # Create evaluator
        evaluator = LocalModelEvaluator(
            model_path=args.model_path,
            config=config,
            hf_path=args.hf_path,
            device=args.device,
            batch_size=args.batch_size,
            verbose=args.verbose,
            debug=args.debug,
        )

        # Run evaluation
        results = evaluator.evaluate_all()

        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(config.output_dir) / f"local_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save results
        results_path = output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {results_path}")
        return 0

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        if args.debug:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
