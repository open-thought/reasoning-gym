import argparse
import asyncio
import json
import logging
import os
from dataclasses import asdict
from datetime import datetime
from typing import Any

import aiohttp
from eval_config import EvalConfig
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

import reasoning_gym
from reasoning_gym.utils import extract_answer


class OpenRouterEvaluator:
    def __init__(self, model: str, config: EvalConfig):
        self.logger = logging.getLogger(f"OpenRouterEvaluator.{model}")
        self.config = config
        self.output_dir = f"{config.eval_dir}/{config.category}"
        os.makedirs(self.output_dir, exist_ok=True)
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": os.getenv("OR_SITE_URL", "localhost"),
            "X-Title": os.getenv("OR_APP_NAME", "Model Evaluation"),
            "Content-Type": "application/json",
        }
        self.semaphore = asyncio.Semaphore(10)  # Control concurrency

    def save_results(self, results: list[dict[str, Any]], dataset, dataset_name) -> dict[str, Any]:
        file_name = f"{self.output_dir}/{dataset_name}.json"
        total_score = sum(r["score"] for r in results)

        metrics = {
            "dataset_name": dataset_name,
            "model": self.model,
            "size": dataset.size,
            "provider": self.config.provider,
            "average_score": total_score / len(results) if results else 0,
            "total_examples": len(results),
            "timestamp": datetime.now().isoformat(),
            "config": asdict(dataset.config),
            "results": results,
        }

        with open(file_name, "w") as f:
            json.dump(metrics, f, indent=2)
        return metrics

    async def get_model_response(self, session: aiohttp.ClientSession, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": self.config.developer_role, "content": self.config.developer_prompt},
                {"role": "user", "content": prompt},
            ],
            "provider": {"order": ["Nebius"], "allow_fallbacks": False},
        }

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(20),
            wait=wait_exponential(multiplier=1, min=1, max=60),
            retry=retry_if_exception_type(
                (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError, ValueError)
            ),
        ):
            with attempt:
                async with session.post(self.base_url, json=payload) as response:
                    data = await response.json()

                    if not data:
                        raise ValueError("Empty response")

                    if not data.get("choices"):
                        raise ValueError("Missing choices in response")

                    return data["choices"][0]["message"]["content"]

        raise Exception("Failed to get valid response after retries")

    async def process_entry(self, session: aiohttp.ClientSession, dataset: Any, entry: Any) -> dict[str, Any]:
        """Process a single entry with concurrency control."""
        async with self.semaphore:
            response = await self.get_model_response(session, entry["question"])
            model_answer = extract_answer(response)
            score = dataset.score_answer(answer=model_answer, entry=entry)

            return {
                "question": entry["question"],
                "expected_answer": str(entry["answer"]),
                "model_answer": model_answer,
                "full_model_response": response,
                "score": score,
                "metadata": str(entry["metadata"]),
            }

    async def evaluate_dataset(self, session: aiohttp.ClientSession, dataset_name: str) -> dict[str, Any]:
        """Evaluate a single dataset asynchronously."""
        self.logger.info(f"\nEvaluating dataset: {dataset_name}")
        dataset = reasoning_gym.create_dataset(
            dataset_name, size=self.config.dataset_size, seed=self.config.dataset_seed
        )

        tasks = [self.process_entry(session, dataset, entry) for entry in dataset]
        results = await asyncio.gather(*tasks)
        return self.save_results(results, dataset, dataset_name)

    async def evaluate_datasets(self) -> list[dict[str, Any]]:
        """Main async evaluation entry point."""
        all_results = []
        async with aiohttp.ClientSession(headers=self.headers) as session:
            return await asyncio.gather(*(self.evaluate_dataset(session, name) for name in self.config.datasets))


async def async_main():
    parser = argparse.ArgumentParser(description="Evaluate models on reasoning datasets")
    parser.add_argument("--yaml", required=True, help="Path to YAML configuration file")
    args = parser.parse_args()

    config = EvalConfig.from_yaml(args.yaml)
    evaluator = OpenRouterEvaluator(model=config.model, config=config)
    results = await evaluator.evaluate_datasets()

    output_dir = f"{config.eval_dir}/{config.category}"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/summary.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    asyncio.run(async_main())
