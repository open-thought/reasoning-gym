from dataclasses import dataclass
from typing import Optional


@dataclass
class ScriptArguments:
    """
    Arguments for the training script.
    """

    dataset_name: str
    dataset_config: Optional[str] = None
    dataset_train_split: str = "train"
    dataset_test_split: str = "test"
    gradient_checkpointing_use_reentrant: bool = False
    ignore_bias_buffers: bool = False
    train_size: int = 1000
    eval_size: int = 100
