"""Configuration classes for the evaluation script"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import yaml

from reasoning_gym.utils import SYSTEM_PROMPTS


def is_valid_unix_filename(filename: str) -> bool:
    """
    Check for shell-safe filenames.
    Only allows alphanumeric characters, hyphens, and underscores.
    """
    if not filename:
        return False
    return bool(re.match(r"^[a-zA-Z0-9_-]+$", filename))


@dataclass
class DatasetConfig:
    """Configuration for a specific dataset"""

    dataset: str
    size: int = 500
    seed: Optional[int] = None
    # Allow any additional dataset-specific parameters
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class CategoryConfig:
    """Configuration for a category of datasets"""

    category: str
    datasets: list[DatasetConfig]


@dataclass
class EvalConfig:
    """Global evaluation configuration"""

    model: str
    provider: Optional[str] = None
    system_prompt: Optional[str] = None
    system_prompt_id: Optional[str] = None
    system_role: str = "system"
    output_dir: str = "results"
    max_concurrent: int = 10
    default_size: int = 500
    default_seed: Optional[int] = None
    save_metadata: bool = False
    save_full_results: bool = False
    # Sampling parameters
    max_tokens: Optional[int] = 32768
    temperature: Optional[float] = 0.6
    top_p: Optional[float] = 0.95
    categories: list[CategoryConfig] = field(default_factory=list)
    
    def get_system_prompt(self) -> str:
        """Get the system prompt to use for evaluation.
        
        Returns:
            The system prompt string to use
        """
        if self.system_prompt is not None and self.system_prompt_id is not None:
            import logging
            logging.warning(
                "Both system_prompt and system_prompt_id are specified in the configuration. "
                "Using system_prompt and ignoring system_prompt_id."
            )
            return self.system_prompt
        
        if self.system_prompt is not None:
            return self.system_prompt
        
        if self.system_prompt_id is not None:
            if self.system_prompt_id in SYSTEM_PROMPTS:
                return SYSTEM_PROMPTS[self.system_prompt_id]
            else:
                import logging
                logging.warning(
                    f"System prompt ID '{self.system_prompt_id}' not found in SYSTEM_PROMPTS. "
                    f"Using default system prompt instead."
                )
        
        # Default case: use the default system prompt
        return SYSTEM_PROMPTS["default"]

    @classmethod
    def from_json(cls, json_path: str) -> "EvalConfig":
        """Load configuration from JSON file"""
        with open(json_path, "r") as f:
            config_data = json.load(f)

        return cls._process_config_data(config_data)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "EvalConfig":
        """Load configuration from YAML file"""
        with open(yaml_path, "r") as f:
            config_data = yaml.safe_load(f)

        return cls._process_config_data(config_data)

    @classmethod
    def _process_config_data(cls, config_data: dict[str, Any]) -> "EvalConfig":
        """Process configuration data from either JSON or YAML"""
        # Extract categories
        categories_data = config_data.pop("categories", [])
        categories = []

        for category_data in categories_data:
            category_name = category_data.get("category")
            if not is_valid_unix_filename(category_name):
                raise ValueError(
                    f"Invalid category name '{category_name}'. Category names must be valid Unix filenames."
                )

            # Process datasets in this category
            datasets_data = category_data.get("datasets", [])
            datasets = []

            for dataset_data in datasets_data:
                # If it's just a string, convert to dict with name
                if isinstance(dataset_data, str):
                    dataset_data = {"name": dataset_data}

                # Extract dataset name
                dataset_name = dataset_data.get("dataset")

                # Extract size and seed with defaults
                size = dataset_data.get("size", config_data.get("default_size", 500))
                seed = dataset_data.get("seed", config_data.get("default_seed"))

                # Extract all other parameters (everything except dataset, size, and seed)
                # If there's a nested 'params' dictionary, use its contents directly
                params = {}
                for k, v in dataset_data.items():
                    if k not in ["dataset", "size", "seed"]:
                        if k == "params" and isinstance(v, dict):
                            # Flatten nested params dictionary
                            params.update(v)
                        else:
                            params[k] = v

                # Create dataset config
                dataset_config = DatasetConfig(
                    dataset=dataset_name,
                    size=size,
                    seed=seed,
                    params=params,
                )
                datasets.append(dataset_config)

            # Create category config
            category_config = CategoryConfig(category=category_name, datasets=datasets)
            categories.append(category_config)

        # Create main config
        return cls(
            model=config_data.get("model"),
            provider=config_data.get("provider", "openai"),
            system_prompt=config_data.get("system_prompt"),
            system_prompt_id=config_data.get("system_prompt_id"),
            system_role=config_data.get("system_role", "system"),
            output_dir=config_data.get("output_dir", "results"),
            max_concurrent=config_data.get("max_concurrent", 10),
            save_metadata=config_data.get("save_metadata", False),
            save_full_results=config_data.get("save_full_results", False),
            # Sampling parameters
            max_tokens=config_data.get("max_tokens", 32768),
            temperature=config_data.get("temperature", 0.6),
            top_p=config_data.get("top_p", 0.95),
            categories=categories,
        )
