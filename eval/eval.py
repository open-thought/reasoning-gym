#!/usr/bin/env python
"""
Evaluation script for reasoning gym datasets.

This script evaluates LLM performance on reasoning gym datasets using the OpenRouter API.

Usage:
    python eval.py --config config.yaml [options]

Options:
    --model MODEL             Override model specified in config
    --output-dir DIR          Override output directory specified in config
    --category CATEGORY       Evaluate only datasets from this category
    --max-concurrent NUM      Maximum number of concurrent API calls
    --n NUM                   Number of completions to generate per prompt (default: 1, each completion is a separate API call)
    --base-url URL            API base URL (default: https://openrouter.ai/api/v1)
    --save-metadata           Save entry metadata in results
    --full-results            Save the full results file
    --verbose                 Print detailed model responses
    --debug                   Enable debug logging
    --resume DIR              Resume evaluation from the specified directory

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
from typing import Any, Optional

from eval_config import CategoryConfig, DatasetConfig, EvalConfig
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

import reasoning_gym
from reasoning_gym.utils import extract_answer


class CheckpointManager:
    """Manages checkpoints for resumable evaluation."""

    def __init__(self, output_dir: Path):
        """Initialize the checkpoint manager.

        Args:
            output_dir: Directory where checkpoints and results are stored
        """
        self.output_dir = output_dir
        self.checkpoint_path = output_dir / "checkpoint.json"
        self.completed_datasets = set()
        self.previous_category_results = {}  # Store previously completed category results
        self.load_checkpoint()

    def load_checkpoint(self) -> None:
        """Load existing checkpoint and previous results if available."""
        # Load checkpoint file
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path, "r") as f:
                checkpoint_data = json.load(f)
                self.completed_datasets = set(checkpoint_data.get("completed_datasets", []))

        # Load previous category results
        for category_dir in self.output_dir.iterdir():
            if category_dir.is_dir():
                category_name = category_dir.name
                self.previous_category_results[category_name] = []

                # Load each dataset result file in this category
                for dataset_file in category_dir.glob("*.json"):
                    try:
                        with open(dataset_file, "r") as f:
                            dataset_result = json.load(f)
                            self.previous_category_results[category_name].append(dataset_result)
                    except Exception as e:
                        logging.warning(f"Error loading previous result {dataset_file}: {str(e)}")

    def is_dataset_completed(self, category: str, dataset: str) -> bool:
        """Check if a dataset has been completed.

        Args:
            category: Category name
            dataset: Dataset name

        Returns:
            True if the dataset has been completed, False otherwise
        """
        return f"{category}/{dataset}" in self.completed_datasets

    def mark_dataset_completed(self, category: str, dataset: str) -> None:
        """Mark a dataset as completed and update checkpoint file.

        Args:
            category: Category name
            dataset: Dataset name
        """
        self.completed_datasets.add(f"{category}/{dataset}")
        self._save_checkpoint()

    def get_dataset_result(self, category: str, dataset: str) -> Optional[dict[str, Any]]:
        """Get previously completed dataset result if available.

        Args:
            category: Category name
            dataset: Dataset name

        Returns:
            Dataset result dict if found, None otherwise
        """
        # Try to find the dataset in previously loaded results first
        if category in self.previous_category_results:
            for dataset_result in self.previous_category_results[category]:
                if dataset_result["name"] == dataset:
                    return dataset_result

        # If not found in memory, try to load from file
        dataset_path = self.output_dir / category / f"{dataset}.json"
        if dataset_path.exists():
            try:
                with open(dataset_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logging.error(f"Error loading dataset result from {dataset_path}: {str(e)}")

        return None

    def _save_checkpoint(self) -> None:
        """Save checkpoint to disk."""
        checkpoint_data = {
            "completed_datasets": list(self.completed_datasets),
            "last_updated": datetime.now().isoformat(),
        }
        with open(self.checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)


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

    def __init__(
        self,
        config: EvalConfig,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        verbose: bool = False,
        debug: bool = False,
        timeout: int = 600,
    ):
        """Initialize the evaluator with configuration.

        Args:
            config: Evaluation configuration
            api_key: API key for the service (optional for some APIs)
            base_url: API base URL
            verbose: Whether to print detailed model responses
            debug: Whether to enable debug logging
        """
        self.config = config
        self.base_url = base_url
        self.verbose = verbose
        self.debug = debug
        self.timeout = timeout

        # Set up logging
        self.logger = logging.getLogger("AsyncModelEvaluator")
        if debug:
            self.logger.setLevel(logging.DEBUG)
            # Enable httpx logs in debug mode
            logging.getLogger("httpx").setLevel(logging.INFO)
        else:
            # Suppress httpx logs in normal mode
            logging.getLogger("httpx").setLevel(logging.WARNING)

        # Set up API client
        self.client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, timeout=self.timeout)

        # Concurrency control
        self.semaphore = asyncio.Semaphore(config.max_concurrent)

        # Metadata
        self.git_hash = get_git_hash()
        self.start_time = datetime.now()

        # Checkpoint and resume related attributes
        self.resume_dir = None
        self.output_dir = None
        self.checkpoint_manager = None

    def create_output_dir(self) -> Path:
        """Create output directory or use existing one for resuming.

        Returns:
            Path to the output directory
        """
        # Check if we're resuming from a previous run
        if self.resume_dir:
            output_dir = Path(self.resume_dir)
            if not output_dir.exists():
                raise ValueError(f"Resume directory {output_dir} does not exist")

            self.logger.info(f"Resuming evaluation from {output_dir}")
            return output_dir

        # Create new output directory
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        model_name = self.config.model.replace("/", "_")

        if len(self.config.categories) == 1:
            # Include category name in the output directory when evaluating a single category
            category_name = self.config.categories[0].category
            output_dir = Path(self.config.output_dir) / f"{model_name}_{category_name}_{timestamp}"
        else:
            # Original format for multiple categories
            output_dir = Path(self.config.output_dir) / f"{model_name}_{timestamp}"

        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _save_dataset_results(self, category_name: str, dataset_name: str, results: dict[str, Any]) -> None:
        """Save individual dataset results to file.

        Args:
            category_name: Category name
            dataset_name: Dataset name
            results: Dataset evaluation results
        """
        category_dir = self.output_dir / category_name
        category_dir.mkdir(exist_ok=True)

        dataset_path = category_dir / f"{dataset_name}.json"
        with open(dataset_path, "w") as f:
            json.dump(results, f, indent=2)

    def _update_partial_summary(self, category_results: list[dict[str, Any]]) -> None:
        """Update partial summary after each category completes.

        Args:
            category_results: List of category results completed so far
        """
        partial_results = {
            "metadata": {
                "timestamp": self.start_time.isoformat(),
                "model": self.config.model,
                "provider": self.config.provider,
                "git_hash": self.git_hash,
                "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "partial": True,
            },
            "categories": category_results,
        }

        # Generate partial summary
        partial_results["summary"] = self.generate_summary(partial_results)

        # Save partial summary
        summary_path = self.output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(partial_results["summary"], f, indent=2)

    async def get_single_response(self, prompt: str) -> str:
        """Get a single response from model with retry logic via OpenRouter.

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
                            {"role": self.config.system_role, "content": self.config.get_system_prompt()},
                            {"role": "user", "content": prompt},
                        ],
                    }

                    # Add sampling parameters if specified
                    if self.config.max_tokens is not None:
                        params["max_tokens"] = self.config.max_tokens
                    if self.config.temperature is not None:
                        params["temperature"] = self.config.temperature
                    if self.config.top_p is not None:
                        params["top_p"] = self.config.top_p

                    # Add provider configuration if specified
                    if self.config.provider:
                        params["extra_body"] = {"provider": {"order": [self.config.provider], "allow_fallbacks": False}}

                    completion = await self.client.chat.completions.create(**params)
                    response = completion.choices[0].message.content

                    if self.verbose:
                        self.logger.info(f"Response: {response}")

                    return response

            except Exception as e:
                delay = min(max_delay, base_delay * (backoff_factor**attempt))
                self.logger.warning(f"Attempt {attempt+1}/{max_retries} failed: {str(e)}")
                self.logger.warning(f"Retrying in {delay:.2f} seconds...")
                await asyncio.sleep(delay)

        raise Exception(f"Failed to get model response after {max_retries} attempts")

    async def get_model_response(self, prompt: str) -> list[str]:
        """Get multiple responses from model by making multiple API calls.

        Args:
            prompt: The prompt to send to the model

        Returns:
            A list of model response texts

        Raises:
            Exception: If all attempts fail
        """
        if self.verbose:
            self.logger.info(f"Prompt: {prompt}")
            self.logger.info(f"Generating {self.config.completions_per_prompt} completions...")

        # Create tasks for multiple completions
        tasks = []
        for i in range(self.config.completions_per_prompt):
            tasks.append(self.get_single_response(prompt))

        # Execute all tasks concurrently
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        valid_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                self.logger.error(f"Completion {i+1} failed: {str(response)}")
            else:
                valid_responses.append(response)
                if self.verbose:
                    self.logger.info(f"Response {len(valid_responses)}: {response}")

        if not valid_responses:
            raise Exception("All completion attempts failed")

        return valid_responses

    async def process_entry(
        self,
        dataset: reasoning_gym.dataset.ProceduralDataset,
        entry: dict[str, Any],
        entry_index: int,
        dataset_name: str,
    ) -> dict[str, Any]:
        """Process a single dataset entry.

        Args:
            dataset: The dataset instance
            entry: The entry to process
            entry_index: Index of the entry in the dataset
            dataset_name: Name of the dataset

        Returns:
            Dict with processing results
        """
        responses = None
        completion_results = []
        best_score = 0.0
        total_score = 0.0
        best_answer = None
        best_response = None

        try:
            # Get multiple model responses
            responses = await self.get_model_response(entry["question"])

            # Count total completions for mean score calculation
            total_completions = len(responses)

            for i, response in enumerate(responses):
                try:
                    # Try to extract answer and score it
                    model_answer = extract_answer(response)
                    score = dataset.score_answer(answer=model_answer, entry=entry)

                    completion_result = {
                        "model_answer": model_answer,
                        "full_model_response": response,
                        "score": score,
                    }

                    # Track scores
                    if score > best_score:
                        best_score = score
                        best_answer = model_answer
                        best_response = response
                    # If we don't have a best answer yet, use the first non-None answer
                    elif best_answer is None and model_answer is not None:
                        best_answer = model_answer
                        best_response = response
                        best_score = score

                    total_score += score

                    completion_results.append(completion_result)

                    if self.verbose:
                        print(f"Question: {entry['question']}")
                        print(f"Expected: {entry['answer']}")
                        print(f"Completion {i+1} Answer: {model_answer}")
                        print(f"Completion {i+1} Score: {score}")
                        print("-" * 40)

                except Exception as e:
                    self.logger.error(f"Error processing completion {i+1}: {str(e)}")
                    # Add failed completion with score 0.0 (already counted in total_completions)
                    completion_results.append(
                        {
                            "model_answer": "ERROR",
                            "full_model_response": response,
                            "score": 0.0,
                            "error": str(e),
                        }
                    )

            # If we have no valid completions, log a warning instead of raising an exception
            if not best_answer:
                self.logger.warning(
                    f"Failed to extract a valid answer from model responses for dataset '{dataset_name}', entry index {entry_index}"
                )
                # Use None instead of empty string as the best answer
                best_answer = None
                best_response = responses[0] if responses and len(responses) > 0 else None
                best_score = 0.0

            # Calculate mean score - count all completions including failures
            mean_score = total_score / total_completions if total_completions > 0 else 0.0

            result = {
                "question": entry["question"],
                "expected_answer": str(entry["answer"]),
                "best_model_answer": best_answer,
                "best_full_model_response": best_response,
                "best_score": best_score,
                "mean_score": mean_score,
                "completions": completion_results,
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
                "best_model_answer": None,
                # First check if we already have a best_response from partial processing
                # If not, then fall back to the first response or None
                "best_full_model_response": (
                    best_response
                    if best_response is not None
                    else (responses[0] if responses and len(responses) > 0 else None)
                ),
                "best_score": best_score if best_score > 0 else 0.0,
                "mean_score": total_score / total_completions if total_completions > 0 else 0.0,
                "error": str(e),
                "completions": completion_results if "completion_results" in locals() else [],
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

        # Check if this dataset has already been completed
        if self.checkpoint_manager.is_dataset_completed(category_name, dataset_name):
            # Get the dataset result from checkpoint manager
            dataset_result = self.checkpoint_manager.get_dataset_result(category_name, dataset_name)
            if dataset_result:
                self.logger.info(f"Skipping already completed dataset: {dataset_name}")
                return dataset_result

            # If we can't load the result, we'll need to re-evaluate the dataset
            self.logger.info(f"Re-evaluating dataset: {dataset_name}")
            # Remove from completed datasets so it will be processed
            self.checkpoint_manager.completed_datasets.discard(f"{category_name}/{dataset_name}")

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

            # Process entries with progress bar, passing the entry index and dataset name
            tasks = [self.process_entry(dataset, entry, idx, dataset_name) for idx, entry in enumerate(all_entries)]
            results = await tqdm_asyncio.gather(*tasks, desc=f"Processing {dataset_name}", leave=True)

            # Calculate metrics
            total_best_score = sum(r["best_score"] for r in results)
            total_mean_score = sum(r["mean_score"] for r in results)
            average_best_score = total_best_score / len(results) if results else 0
            average_mean_score = total_mean_score / len(results) if results else 0

            dataset_results = {
                "name": dataset_name,
                "category": category_name,
                "average_best_score": average_best_score,
                "average_mean_score": average_mean_score,
                "total_examples": len(results),
                "config": {"size": dataset_config.size, "seed": dataset_config.seed, **dataset_config.params},
                "system_prompt": self.config.get_system_prompt(),
                "completions_per_prompt": self.config.completions_per_prompt,
                "results": results,
            }

            # Mark dataset as completed and save results
            self.checkpoint_manager.mark_dataset_completed(category_name, dataset_name)
            self._save_dataset_results(category_name, dataset_name, dataset_results)

            return dataset_results

        except Exception as e:
            self.logger.error(f"Error evaluating dataset {dataset_name}: {str(e)}")
            return {
                "name": dataset_name,
                "category": category_name,
                "average_best_score": 0.0,
                "average_mean_score": 0.0,
                "total_examples": 0,
                "config": {"size": dataset_config.size, "seed": dataset_config.seed, **dataset_config.params},
                "system_prompt": self.config.get_system_prompt(),
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

        # Check if all datasets in this category are already completed
        all_completed = True
        for dataset_config in category_config.datasets:
            if not self.checkpoint_manager.is_dataset_completed(category_name, dataset_config.dataset):
                all_completed = False
                break

        # If all datasets are completed and we have previous results, use them
        if all_completed and category_name in self.checkpoint_manager.previous_category_results:
            self.logger.info(f"Using previously completed results for category: {category_name}")
            return {
                "name": category_name,
                "datasets": self.checkpoint_manager.previous_category_results[category_name],
            }

        # Process datasets sequentially to ensure proper checkpointing
        dataset_results = []
        for dataset_config in category_config.datasets:
            result = await self.evaluate_dataset(category_name, dataset_config)
            dataset_results.append(result)

        return {
            "name": category_name,
            "datasets": dataset_results,
        }

    async def evaluate_all(self) -> dict[str, Any]:
        """Evaluate all categories and datasets, resuming from checkpoint if available.

        Returns:
            Dict with all evaluation results and summary
        """
        self.logger.info(f"Starting evaluation of {len(self.config.categories)} categories")

        # Initialize output directory and checkpoint manager
        self.output_dir = self.create_output_dir()
        self.checkpoint_manager = CheckpointManager(self.output_dir)

        # Process each category sequentially to ensure proper checkpointing
        category_results = []
        for category in self.config.categories:
            category_result = await self.evaluate_category(category)
            category_results.append(category_result)

            # Update partial summary after each category
            self._update_partial_summary(category_results)

        # Generate results structure
        results = {
            "metadata": {
                "timestamp": self.start_time.isoformat(),
                "model": self.config.model,
                "provider": self.config.provider,
                "git_hash": self.git_hash,
                "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "partial": False,  # Mark as complete
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
            "dataset_best_scores": {},
            "dataset_mean_scores": {},
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
                                summary["dataset_best_scores"][dataset_name] = dataset["average_best_score"]
                                summary["dataset_mean_scores"][dataset_name] = dataset["average_mean_score"]
                                summary["total_datasets"] += 1
                                summary["total_examples"] += dataset["total_examples"]
                                dataset_found = True
                                break

                # If dataset wasn't found in results (error), add with score 0
                if not dataset_found:
                    summary["dataset_best_scores"][dataset_name] = 0.0
                    summary["dataset_mean_scores"][dataset_name] = 0.0
                    summary["total_datasets"] += 1

        return summary

    def save_results(self, results: dict[str, Any]) -> tuple[str, str]:
        """Save evaluation results to files.

        Args:
            results: The evaluation results to save

        Returns:
            Tuple of (results_path, summary_path)
        """
        # Output directory is already created during evaluation
        results_path = None

        # Save full results if configured to do so
        if self.config.save_full_results:
            results_path = self.output_dir / "results.json"
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)

        # Add timestamp, git hash, model, provider, sampling parameters, and duration to summary
        summary_data = results["summary"].copy()
        summary_data["timestamp"] = self.start_time.isoformat()
        summary_data["git_hash"] = self.git_hash
        summary_data["model"] = self.config.model
        summary_data["provider"] = self.config.provider
        summary_data["system_prompt"] = self.config.get_system_prompt()
        if self.config.system_prompt_id:
            summary_data["system_prompt_id"] = self.config.system_prompt_id
        summary_data["max_tokens"] = self.config.max_tokens
        summary_data["temperature"] = self.config.temperature
        summary_data["top_p"] = self.config.top_p
        summary_data["completions_per_prompt"] = self.config.completions_per_prompt
        summary_data["duration_seconds"] = results["metadata"]["duration_seconds"]
        summary_data["partial"] = False  # Mark as complete

        # Save summary
        summary_path = self.output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary_data, f, indent=2)

        # Individual dataset results are already saved during evaluation

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
        system_prompt = self.config.get_system_prompt()
        print(f"System Prompt: {system_prompt[:50]}..." if len(system_prompt) > 50 else system_prompt)
        print(f"Max Tokens: {self.config.max_tokens}")
        print(f"Temperature: {self.config.temperature}")
        print(f"Top-p: {self.config.top_p}")
        print(f"Completions per prompt: {self.config.completions_per_prompt}")
        print(f"Git Hash: {self.git_hash}")
        print(f"Duration: {results['metadata']['duration_seconds']:.2f} seconds")
        print()

        print("Dataset Scores (in configuration order):")
        print("  Dataset Name                  Best Score    Mean Score    Examples")
        print("  ------------------------------------------------------------------")
        for dataset_name in summary["dataset_best_scores"].keys():
            best_score = summary["dataset_best_scores"][dataset_name]
            mean_score = summary["dataset_mean_scores"][dataset_name]

            # Find the number of examples for this dataset
            examples = 0
            for category in results["categories"]:
                for dataset in category["datasets"]:
                    if dataset["name"] == dataset_name:
                        examples = dataset["total_examples"]
                        break

            # Use fixed-width formatting for better alignment
            print(f"  {dataset_name:<30} {best_score:>8.1%}    {mean_score:>8.1%}    {examples:>8}")

        print()
        print(f"Total datasets: {summary['total_datasets']}")
        print(f"Total examples: {summary['total_examples']}")


async def main_async():
    """Main async function."""
    parser = argparse.ArgumentParser(description="Evaluate models on reasoning datasets")
    parser.add_argument("--config", required=True, help="Path to configuration file (YAML or JSON)")
    parser.add_argument("--model", help="Override model specified in config")
    parser.add_argument("--output-dir", help="Override output directory specified in config")
    parser.add_argument("--category", help="Evaluate only datasets from this category")
    parser.add_argument("--max-concurrent", type=int, help="Maximum number of concurrent API calls")
    parser.add_argument("--n", type=int, default=1, help="Number of completions to generate per prompt")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1", help="API base URL")
    parser.add_argument(
        "--api-key",
        help="API key for the service (optional for some APIs, defaults to OPENROUTER_API_KEY env var for OpenRouter URLs)",
    )
    parser.add_argument("--save-metadata", action="store_true", help="Save entry metadata in results")
    parser.add_argument("--full-results", action="store_true", help="Save the full results file")
    parser.add_argument("--verbose", action="store_true", help="Print detailed model responses")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--resume", help="Resume evaluation from the specified directory")
    parser.add_argument("--timeout", type=int, default=600, help="Time in seconds for timeout of a single request")

    args = parser.parse_args()

    # Get API key from command line or environment variable
    api_key = args.api_key
    if api_key is None:
        # If base_url is OpenRouter, try to get API key from environment
        if args.base_url.startswith("https://openrouter.ai/api"):
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                print("Warning: OPENROUTER_API_KEY environment variable is not set")
                print("Please set it using: export OPENROUTER_API_KEY=your-api-key")
                print("Or provide it directly with --api-key")
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
    if args.n:
        config.completions_per_prompt = args.n
    if args.save_metadata:
        config.save_metadata = True
    if args.full_results:
        config.save_full_results = True

    # Filter categories if --category is specified
    if args.category:
        # Keep only the specified category
        filtered_categories = [cat for cat in config.categories if cat.category == args.category]
        if not filtered_categories:
            print(f"Error: Category '{args.category}' not found in configuration")
            return 1
        config.categories = filtered_categories

    # Create evaluator
    evaluator = AsyncModelEvaluator(
        config=config,
        api_key=api_key,
        base_url=args.base_url,
        verbose=args.verbose,
        debug=args.debug,
        timeout=args.timeout,
    )

    # Set resume directory if specified
    if args.resume:
        evaluator.resume_dir = args.resume

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
