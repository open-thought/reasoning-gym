import math
import re
from typing import Any, Callable, Dict


class RewardRegistry:
    """Simple registry for secondary reward functions."""

    def __init__(self):
        self.reward_functions = {}

    def register(self, name: str):
        """Register a reward function."""

        def decorator(func):
            self.reward_functions[name] = func
            return func

        return decorator

    def get(self, name: str):
        """Get a reward function by name."""
        return self.reward_functions.get(name)

    def list_functions(self):
        """List available reward function names."""
        return list(self.reward_functions.keys())


reward_registry = RewardRegistry()


@reward_registry.register("cosine")
def cosine_scaled_reward(solution_str, scaling_factor, **kwargs):
    """Reward function that scales based on completion length using a cosine schedule."""
    min_value_wrong = -1.0
    max_value_wrong = -0.5
    min_value_correct = 0.5
    max_value_correct = 1.0
    max_len = 1000

    is_correct = kwargs.get("is_correct", False)
    gen_len = len(solution_str)

    # Apply cosine scaling based on length
    progress = gen_len / max_len
    cosine = math.cos(progress * math.pi)

    if is_correct:
        min_value = min_value_correct
        max_value = max_value_correct
    else:
        min_value = max_value_wrong
        max_value = min_value_wrong

    cosine_scaled_reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
    return cosine_scaled_reward * scaling_factor


@reward_registry.register("format")
def compute_format_reward(solution_str: str, scaling_factor: float = 0.2, **kwargs) -> float:
    """Reward use of exactly one correctly structured <think> and <answer> block."""
    pattern = r"\s*<think>.*?</think>\s*<answer>.*?</answer>"
    if not re.match(pattern, solution_str, re.DOTALL):
        return 0.0
    think_matches = list(re.finditer(r"<think>(.*?)</think>", solution_str, re.DOTALL))
    answer_matches = list(re.finditer(r"<answer>(.*?)</answer>", solution_str, re.DOTALL))
    if len(think_matches) != 1 or len(answer_matches) != 1:
        return 0.0
    think_content = think_matches[0].group(1)
    if "<think>" in think_content or "<answer>" in think_content:
        return 0.0
    answer_content = answer_matches[0].group(1)
    if "<answer>" in answer_content or "<think>" in answer_content:
        return 0.0
    return 1.0 * scaling_factor


@reward_registry.register("length")
def length_reward(solution_str, correctness_score, scaling_factor, **kwargs):
    """Reward length appropriately based on correctness."""
    epsilon = 1e-6
    max_score = kwargs.get("max_score", 1.0)
    max_output_length = kwargs.get("max_output_length", 1024)

    generation_len = len(solution_str)
    progress = min(generation_len / max_output_length, 1.0)

    if correctness_score < max_score - epsilon:
        length_reward = (max_score - correctness_score) * progress
    else:
        length_reward = -progress

    return length_reward * scaling_factor
