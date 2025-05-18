from dataclasses import dataclass, field
from typing import Literal


# check ./recipes/MODEL_NAME/PT_METHOD/model_DATASET.yaml
@dataclass
class ModelPTConfig:
    # //*******Model post-training configs*******//
    model_post_train_type: Literal["grpo", "sft"] = field(default="grpo")
    model_post_train_dataset_name: str = field(default="curated_deepscaler")
    model_post_train_dataset_config: str | None = field(default=None)

    rl_post_train_reward_funcs: list[str] = field(default_factory=lambda: ["accuracy", "format"])
    rl_post_train_reward_weights: list[str] = field(default_factory=lambda: [2.0, 1.0])
    cosine_min_value_wrong: float = field(default=0.0)
    cosine_max_value_wrong: float = field(default=-0.5)
    cosine_min_value_correct: float = field(default=0.5)
    cosine_max_value_correct: float = field(default=1.0)
    cosine_max_len: int = field(default=1000)
    repetition_n_grams: int = field(default=3)
    repetition_max_penalty: float = field(default=-1.0)
