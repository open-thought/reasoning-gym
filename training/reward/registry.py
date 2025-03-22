from typing import Dict, Callable, Any
import re

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
    max_score = kwargs.get('max_score', 1.0)
    max_output_length = kwargs.get('max_output_length', 1024)
    
    generation_len = len(solution_str)
    progress = min(generation_len / max_output_length, 1.0)
    
    if correctness_score < max_score - epsilon:
        length_reward = (max_score - correctness_score) * progress
    else:
        length_reward = -progress
    
    return length_reward * scaling_factor
    
    