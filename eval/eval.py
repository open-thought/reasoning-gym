import argparse
import asyncio
import json
import os
import re
import time
from datetime import datetime
from typing import Any

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

from reasoning_gym.factory import create_dataset
from reasoning_gym.utils import SYSTEM_PROMPTS


class AsyncOpenRouterEvaluator:
    def __init__(self, model: str, max_concurrent: int = 10):
        self.client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))
        self.model = model
        self.extra_headers = {}
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def get_model_response(self, prompt: str) -> str:
        """Get response from the model via OpenRouter API with rate limiting."""
        async with self.semaphore:
            try:
                completion = await self.client.chat.completions.create(
                    extra_headers=self.extra_headers,
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPTS["default"]},
                        {"role": "user", "content": prompt},
                    ],
                )
                return completion.choices[0].message.content
            except Exception as e:
                print(f"Error calling OpenRouter API: {str(e)}")
                raise

    def parse_model_response(self, response: str) -> str:
        """Gather the final answer between the <answer> and </answer> tags."""
        match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        return match.group(1).strip() if match else response

    async def process_single_question(self, entry: dict, dataset) -> dict:
        """Process a single question and return the result."""
        response = await self.get_model_response(entry["question"])
        answer = self.parse_model_response(response)
        score = dataset.score_answer(answer=answer, entry=entry)

        return {
            "question": entry["question"],
            "expected_answer": entry["answer"],
            "model_answer": answer,
            "full_model_response": response,
            "score": score,
            "metadata": entry["metadata"],
        }

    async def evaluate_dataset(self, dataset_config: dict[str, Any]) -> dict[str, Any]:
        """Evaluate a single dataset with concurrent question processing."""
        dataset_name = dataset_config.pop("name")
        print(f"\nEvaluating dataset: {dataset_name}")

        try:
            # Create dataset with its specific configuration
            data = create_dataset(dataset_name, **dataset_config)
            all_entries = list(data)

            # Process all questions concurrently
            tasks = [self.process_single_question(entry, data) for entry in all_entries]

            # Use tqdm to track progress
            results = await tqdm_asyncio.gather(*tasks, desc=f"Processing {dataset_name}")

            # Calculate aggregate metrics
            total_score = sum(r["score"] for r in results)
            metrics = {
                "dataset_name": dataset_name,
                "model": self.model,
                "size": len(data),
                "average_score": total_score / len(results) if results else 0,
                "total_examples": len(results),
                "timestamp": datetime.now().isoformat(),
                "config": dataset_config,
            }

            return {"metrics": metrics, "results": results}

        except Exception as e:
            print(f"Error evaluating dataset {dataset_name}: {str(e)}")
            return None

    async def evaluate_datasets(self, dataset_configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Evaluate multiple datasets concurrently."""
        tasks = [self.evaluate_dataset(config) for config in dataset_configs]

        # Process all datasets concurrently
        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]


async def main_async():
    parser = argparse.ArgumentParser(description="Evaluate models on reasoning datasets")
    parser.add_argument("--model", required=True, help="Model to evaluate")
    parser.add_argument("--config", required=True, help="Path to JSON configuration file")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--max-concurrent", type=int, default=10, help="Maximum number of concurrent API calls")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset configurations
    with open(args.config, "r") as f:
        dataset_configs = json.load(f)

    evaluator = AsyncOpenRouterEvaluator(model=args.model, max_concurrent=args.max_concurrent)

    eval_start_time = time.time()
    all_results = await evaluator.evaluate_datasets(dataset_configs)
    print(f"Time taken to collect evaluation data: {time.time() - eval_start_time:.2f} seconds")
    # Save results
    output_file = os.path.join(
        args.output_dir, f"evaluation_{args.model.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Create and save summary
    summary = []
    for result in all_results:
        metrics = result["metrics"]
        summary_entry = {
            "dataset_name": metrics["dataset_name"],
            "model": metrics["model"],
            "average_score": metrics["average_score"],
            "total_examples": metrics["total_examples"],
            "timestamp": metrics["timestamp"],
            "config": metrics["config"],
        }
        summary.append(summary_entry)

    summary_file = os.path.join(
        args.output_dir, f"summary_{args.model.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\nEvaluation Summary:")
    for entry in summary:
        print(f"\nDataset: {entry['dataset_name']}")
        print(f"Average Score: {entry['average_score']:.2%}")
        print(f"Total Examples: {entry['total_examples']}")

    print(f"\nDetailed results saved to: {output_file}")
    print(f"Summary saved to: {summary_file}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
