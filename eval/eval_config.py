import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from typing import Union

import yaml

from reasoning_gym.utils import SYSTEM_PROMPTS


def get_git_commit_hash():
    """Get the current git commit hash."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode("ascii").strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


@dataclass
class EvalConfig:
    category: str
    datasets: Union[str, list[str]]
    eval_dir: str
    dataset_size: int
    dataset_seed: int
    model: str
    provider: str
    developer_role: str = "system"
    developer_prompt: str = SYSTEM_PROMPTS["DeepSeekZero"]

    @classmethod
    def from_yaml(cls, yaml_path: str):
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        return cls(**config)


# Custom JSON encoder to handle datetime objects
class EvalOutputEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)
