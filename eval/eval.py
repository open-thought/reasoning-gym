#!/usr/bin/env python
"""
Evaluation script for reasoning gym datasets.

This script evaluates LLM performance on reasoning gym datasets using the OpenRouter API.

Usage:
    python eval.py --config config.yaml [options]

Options:
    --model MODEL             Override model specified in config
    --output-dir DIR          Override output directory specified in config
    --max-concurrent NUM      Maximum number of concurrent API calls
    --save-metadata           Save entry metadata in results
    --full-results            Save the full results file
    --verbose                 Print detailed model responses
    --debug                   Enable debug logging

Environment variables:
    OPENROUTER_API_KEY        Required API key for OpenRouter
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Union

from eval_config import CategoryConfig, DatasetConfig, EvalConfig
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

import reasoning_gym
from reasoning_gym.utils import extract_answer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

# httpx logging will be configured in the AsyncModelEvaluator class
# based on the debug flag


def get_git_hash() -> str:
    """Get current git hash for reproducibility."""
    cmd = ["git", "rev-parse", "HEAD"]
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.PIPE).strip()
    except Exception:
        return "unknown"


class AsyncModelEvaluator:
    """Evaluates models on reasoning datasets with async API calls via OpenRouter."""

    def __init__(self, config: EvalConfig, verbose: bool = False, debug: bool = False):
        """Initialize the evaluator with configuration.

        Args:
            config: Evaluation configuration
            verbose: Whether to print detailed model responses
            debug: Whether to enable debug logging
        """
        self.config = config
        self.verbose = verbose
        self.debug = debug

        # Set up logging
        self.logger = logging.getLogger("AsyncModelEvaluator")
        if debug:
            self.logger.setLevel(logging.DEBUG)
            # Enable httpx logs in debug mode
            logging.getLogger("httpx").setLevel(logging.INFO)
        else:
            # Suppress httpx logs in normal mode
            logging.getLogger("httpx").setLevel(logging.WARNING)

        # Set up OpenRouter API client
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")

        self.client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

        # Concurrency control
        self.semaphore = asyncio.Semaphore(config.max_concurrent)

        # Metadata
        self.git_hash = get_git_hash()
        self.start_time = datetime.now()

    async def get_model_response(self, prompt: str) -> str:
        """Get response from model with retry logic via OpenRouter.

        Args:
            prompt: The prompt to send to the model

        Returns:
            The model's response text

        Raises:
            Exception: If all retries fail
        """
        max_retries = 10
        base_delay = 1.0
        max_delay = 60.0
        backoff_factor = 2.0

        for attempt in range(max_retries):
            try:
                async with self.semaphore:
                    # Prepare API call parameters
                    params = {
                        "model": self.config.model,
                        "messages": [
                            {"role": self.config.system_role, "content": self.config.system_prompt},
                            {"role": "user", "content": prompt},
                        ],
                    }

                    # Add provider configuration if specified
                    if self.config.provider:
                        params["extra_body"] = {"provider": {"order": [self.config.provider], "allow_fallbacks": False}}

                    completion = await self.client.chat.completions.create(**params)
                    response = completion.choices[0].message.content

                    if self.verbose:
                        self.logger.info(f"Prompt: {prompt}")
                        self.logger.info(f"Response: {response}")

                    return response

            except Exception as e:
                delay = min(max_delay, base_delay * (backoff_factor**attempt))
                self.logger.warning(f"Attempt {attempt+1}/{max_retries} failed: {str(e)}")
                self.logger.warning(f"Retrying in {delay:.2f} seconds...")
                await asyncio.sleep(delay)

        raise Exception(f"Failed to get model response after {max_retries} attempts")

    async def process_entry(
        self, dataset: reasoning_gym.dataset.ProceduralDataset, entry: dict[str, Any]
    ) -> dict[str, Any]:
        """Process a single dataset entry.

        Args:
            dataset: The dataset instance
            entry: The entry to process

        Returns:
            Dict with processing results
        """
        try:
            response = await self.get_model_response(entry["question"])
            model_answer = extract_answer(response)
            score = dataset.score_answer(answer=model_answer, entry=entry)

            if self.verbose:
                print(f"Question: {entry['question']}")
                print(f"Expected: {entry['answer']}")
                print(f"Answer: {model_answer}")
                print(f"Score: {score}")
                print("-" * 40)

            result = {
                "question": entry["question"],
                "expected_answer": str(entry["answer"]),
                "model_answer": model_answer,
                "full_model_response": response,
                "score": score,
            }

            # Only include metadata if configured to do so
            if self.config.save_metadata:
                result["metadata"] = entry["metadata"]

            return result

        except Exception as e:
            self.logger.error(f"Error processing entry: {str(e)}")
            result = {
                "question": entry["question"],
                "expected_answer": str(entry["answer"]),
                "model_answer": "ERROR",
                "full_model_response": f"Error: {str(e)}",
                "score": 0.0,
                "error": str(e),
            }

            # Only include metadata if configured to do so
            if self.config.save_metadata:
                result["metadata"] = entry["metadata"]

            return result

    async def evaluate_dataset(self, category_name: str, dataset_config: DatasetConfig) -> dict[str, Any]:
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
            # Create dataset with all parameters
            dataset_params = {}

            # Add all parameters from the config params dictionary
            # Make sure we don't have a nested 'params' dictionary
            for k, v in dataset_config.params.items():
                if k != "params":
                    dataset_params[k] = v
                elif isinstance(v, dict):
                    # If there's a nested params dict, flatten it
                    dataset_params.update(v)

            # Add size and seed if they're not None
            if dataset_config.size is not None:
                dataset_params["size"] = dataset_config.size
            if dataset_config.seed is not None:
                dataset_params["seed"] = dataset_config.seed

            dataset = reasoning_gym.create_dataset(dataset_name, **dataset_params)

            # Get all entries
            all_entries = list(dataset)

            # Process entries with progress bar
            tasks = [self.process_entry(dataset, entry) for entry in all_entries]
            results = await tqdm_asyncio.gather(*tasks, desc=f"Processing {dataset_name}", leave=True)

            # Calculate metrics
            total_score = sum(r["score"] for r in results)
            average_score = total_score / len(results) if results else 0

            return {
                "name": dataset_name,
                "category": category_name,
                "average_score": average_score,
                "total_examples": len(results),
                "config": {"size": dataset_config.size, "seed": dataset_config.seed, **dataset_config.params},
                "system_prompt": self.config.system_prompt,
                "results": results,
            }

        except Exception as e:
            self.logger.error(f"Error evaluating dataset {dataset_name}: {str(e)}")
            return {
                "name": dataset_name,
                "category": category_name,
                "average_score": 0.0,
                "total_examples": 0,
                "config": {"size": dataset_config.size, "seed": dataset_config.seed, **dataset_config.params},
                "system_prompt": self.config.system_prompt,
                "error": str(e),
                "results": [],
            }

    async def evaluate_category(self, category_config: CategoryConfig) -> dict[str, Any]:
        """Evaluate all datasets in a category.

        Args:
            category_config: Configuration for the category

        Returns:
            Dict with category evaluation results
        """
        category_name = category_config.category
        self.logger.info(f"Evaluating category: {category_name}")

        tasks = [self.evaluate_dataset(category_name, dataset_config) for dataset_config in category_config.datasets]

        dataset_results = await asyncio.gather(*tasks)

        return {
            "name": category_name,
            "datasets": dataset_results,
        }

    async def evaluate_all(self) -> dict[str, Any]:
        """Evaluate all categories and datasets.

        Returns:
            Dict with all evaluation results and summary
        """
        self.logger.info(f"Starting evaluation of {len(self.config.categories)} categories")

        tasks = [self.evaluate_category(category) for category in self.config.categories]
        category_results = await asyncio.gather(*tasks)

        # Generate results structure
        results = {
            "metadata": {
                "timestamp": self.start_time.isoformat(),
                "model": self.config.model,
                "provider": self.config.provider,
                "git_hash": self.git_hash,
                "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
            },
            "categories": category_results,
        }

        # Generate summary
        results["summary"] = self.generate_summary(results)

        return results

    def generate_summary(self, results: dict[str, Any]) -> dict[str, Any]:
        """Generate a summary of evaluation results in the original configuration order.

        Args:
            results: The full evaluation results

        Returns:
            Dict with summary information
        """
        summary = {
            "total_datasets": 0,
            "total_examples": 0,
            "dataset_scores": {},
        }

        # Iterate through categories and datasets in the original order from config
        for category_config in self.config.categories:
            for dataset_config in category_config.datasets:
                dataset_name = dataset_config.dataset
                dataset_found = False

                # Find corresponding results
                for category in results["categories"]:
                    if category["name"] == category_config.category:
                        for dataset in category["datasets"]:
                            if dataset["name"] == dataset_name:
                                # Add to summary in original order
                                summary["dataset_scores"][dataset_name] = dataset["average_score"]
                                summary["total_datasets"] += 1
                                summary["total_examples"] += dataset["total_examples"]
                                dataset_found = True
                                break

                # If dataset wasn't found in results (error), add with score 0
                if not dataset_found:
                    summary["dataset_scores"][dataset_name] = 0.0
                    summary["total_datasets"] += 1

        return summary

    def save_results(self, results: dict[str, Any]) -> tuple[str, str]:
        """Save evaluation results to files.

        Args:
            results: The evaluation results to save

        Returns:
            Tuple of (results_path, summary_path)
        """
        # Create output directory with timestamp
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        model_name = self.config.model.replace("/", "_")

        # Format directory name with model and timestamp only
        output_dir = Path(self.config.output_dir) / f"{model_name}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        results_path = None

        # Save full results if configured to do so
        if self.config.save_full_results:
            results_path = output_dir / "results.json"
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)

        # Add timestamp, git hash, model, provider, and duration to summary
        summary_data = results["summary"].copy()
        summary_data["timestamp"] = self.start_time.isoformat()
        summary_data["git_hash"] = self.git_hash
        summary_data["model"] = self.config.model
        summary_data["provider"] = self.config.provider
        summary_data["system_prompt"] = self.config.system_prompt
        summary_data["duration_seconds"] = results["metadata"]["duration_seconds"]

        # Save summary
        summary_path = output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary_data, f, indent=2)

        # Save individual dataset results
        for category in results["categories"]:
            category_dir = output_dir / category["name"]
            category_dir.mkdir(exist_ok=True)

            for dataset in category["datasets"]:
                dataset_path = category_dir / f"{dataset['name']}.json"
                with open(dataset_path, "w") as f:
                    json.dump(dataset, f, indent=2)

        return str(results_path) if results_path else None, str(summary_path)

    def print_summary(self, results: dict[str, Any]) -> None:
        """Print a summary of evaluation results to the console.

        Args:
            results: The evaluation results
        """
        summary = results["summary"]

        print("\nEvaluation Summary:")
        print("------------------")
        print(f"Model: {self.config.model}")
        print(f"Provider: {self.config.provider}")
        print(
            f"System Prompt: {self.config.system_prompt[:50]}..."
            if len(self.config.system_prompt) > 50
            else self.config.system_prompt
        )
        print(f"Git Hash: {self.git_hash}")
        print(f"Duration: {results['metadata']['duration_seconds']:.2f} seconds")
        print()

        print("Dataset Scores (in configuration order):")
        for dataset_name, score in summary["dataset_scores"].items():
            # Find the number of examples for this dataset
            examples = 0
            for category in results["categories"]:
                for dataset in category["datasets"]:
                    if dataset["name"] == dataset_name:
                        examples = dataset["total_examples"]
                        break

            print(f"  {dataset_name}: {score:.1%}  ({examples} examples)")

        print()
        print(f"Total datasets: {summary['total_datasets']}")
        print(f"Total examples: {summary['total_examples']}")


async def main_async():
    """Main async function."""
    parser = argparse.ArgumentParser(description="Evaluate models on reasoning datasets")
    parser.add_argument("--config", required=True, help="Path to configuration file (YAML or JSON)")
    parser.add_argument("--model", help="Override model specified in config")
    parser.add_argument("--output-dir", help="Override output directory specified in config")
    parser.add_argument("--max-concurrent", type=int, help="Maximum number of concurrent API calls")
    parser.add_argument("--save-metadata", action="store_true", help="Save entry metadata in results")
    parser.add_argument("--full-results", action="store_true", help="Save the full results file")
    parser.add_argument("--verbose", action="store_true", help="Print detailed model responses")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Check for required API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY environment variable is not set")
        print("Please set it using: export OPENROUTER_API_KEY=your-api-key")
        return 1

    # Load configuration
    config_path = args.config
    if config_path.endswith(".yaml") or config_path.endswith(".yml"):
        config = EvalConfig.from_yaml(config_path)
    elif config_path.endswith(".json"):
        config = EvalConfig.from_json(config_path)
    else:
        print("Error: Configuration file must be YAML or JSON")
        return 1

    # Apply command line overrides
    if args.model:
        config.model = args.model
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.max_concurrent:
        config.max_concurrent = args.max_concurrent
    if args.save_metadata:
        config.save_metadata = True
    if args.full_results:
        config.save_full_results = True

    # Create evaluator
    evaluator = AsyncModelEvaluator(config=config, verbose=args.verbose, debug=args.debug)

    # Run evaluation
    try:
        results = await evaluator.evaluate_all()

        # Save and print results
        results_path, summary_path = evaluator.save_results(results)
        evaluator.print_summary(results)

        if results_path:
            print(f"\nResults saved to: {results_path}")
        print(f"Summary saved to: {summary_path}")

        return 0
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        if args.debug:
            import traceback

            traceback.print_exc()
        return 1


def main():
    """Entry point."""
    exit_code = asyncio.run(main_async())
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
