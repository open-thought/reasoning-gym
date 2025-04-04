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



def main():
    argparser = argparse.ArgumentParser(description="Evaluate reasoning gym datasets.")
    argparser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    args = argparser.parse_args()


    config_path = args.config
    if config_path.endswith(".yaml") or config_path.endswith(".yml"):
        config = EvalConfig.from_yaml(config_path)
    elif config_path.endswith(".json"):
        config = EvalConfig.from_json(config_path)
    else:
        print("Error: Configuration file must be YAML or JSON")
        return 1

    for category in config.categories:
        for dataset in category.datasets:
            rg_dataset = reasoning_gym.create_dataset(
                dataset.dataset,
                size=10,
                seed=42,
                **dataset.params
            )
            print(rg_dataset)


if __name__ == "__main__":
    main()