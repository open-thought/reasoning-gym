from dataclasses import dataclass
from typing import Union

import yaml

from reasoning_gym.utils import SYSTEM_PROMPTS


@dataclass
class EvalConfig:
    category: str
    datasets: Union[str, list[str]]
    eval_dir: str
    dataset_size: int
    dataset_seed: int
    model: str = "deepseek/deepseek-r1"
    provider: str = "Nebius"
    developer_role: str = "system"
    developer_prompt: str = SYSTEM_PROMPTS["DeepSeekZero"]

    @classmethod
    def from_yaml(cls, yaml_path: str):
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        return cls(**config)
