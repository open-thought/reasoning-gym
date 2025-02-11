import argparse
import json
import logging
import os
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List

import requests
from eval_config import EvalConfig
from requests.exceptions import RequestException
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

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

    def save_results(self, results: List[Dict[str, Any]], dataset, dataset_name) -> Dict[str, Any]:

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
            "results": results,  # save results to allow for performance recalculation
        }

        with open(file_name, "w") as f:
            json.dump(metrics, f, indent=2)
        return metrics

    def prepare_messages(self, prompt: str) -> List[Dict[str, str]]:
        messages = [
            {"role": self.config.developer_role, "content": self.config.developer_prompt},
            {"role": "user", "content": prompt},
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "provider": {"order": ["Nebius"], "allow_fallbacks": False},
        }  # make sure only one provider is used

        return payload

    @retry(
        retry=retry_if_exception_type(RequestException),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
    )
    def get_model_response(self, prompt: str) -> str:
        """Get response from the model via OpenRouter API."""

        payload = self.prepare_messages(prompt)
        try:
            response = requests.post(self.base_url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RequestException(
                f"API request failed: {str(e)}", {"endpoint": self.base_url, "model": self.model}
            ) from e
        return response.json()["choices"][0]["message"]["content"]

    def evaluate_datasets(self) -> List[Dict[str, Any]]:
        """Evaluate model on multiple datasets with their respective configurations."""
        all_results = []

        for dataset_name in self.config.datasets:
            self.logger.info(f"\nEvaluating dataset: {dataset_name}")

            # Create dataset with its specific configuration
            dataset = reasoning_gym.create_dataset(
                dataset_name, size=self.config.dataset_size, seed=self.config.dataset_seed
            )
            results = []

            for i, entry in enumerate(dataset):
                print(f"On example {i+1} of {len(dataset)}")
                response = self.get_model_response(entry["question"])
                model_answer = extract_answer(response)

                score = dataset.score_answer(answer=model_answer, entry=entry)

                result = {
                    "question": entry["question"],
                    "expected_answer": str(entry["answer"]),
                    "model_answer": model_answer,
                    "score": score,
                    "metadata": str(entry["metadata"]),
                }
                results.append(result)

            metrics = self.save_results(results, dataset, dataset_name)

            all_results.append({"metrics": metrics, "results": results})

        return all_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate models on reasoning datasets")
    parser.add_argument("--yaml", required=True, help="Path to YAML configuration file")

    args = parser.parse_args()
    config = EvalConfig.from_yaml(args.yaml)
    output_dir = f"{config.eval_dir}/{config.category}"
    os.makedirs(output_dir, exist_ok=True)

    evaluator = OpenRouterEvaluator(model=config.model, config=config)
    all_results = evaluator.evaluate_datasets()

    with open(f"{output_dir}/summary.json", "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
