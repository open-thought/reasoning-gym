import re
from typing import Optional

# DeepSeek Zero system prompt
SYSTEM_PROMPTS = {
    "DeepSeekZero": """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>
"""
}


def extract_answer(completion: str, tag_name: str = "answer") -> Optional[str]:
    regex = f"<{tag_name}>(.*?)</{tag_name}>"
    matches = list(
        re.finditer(
            regex,
            completion,
            flags=re.DOTALL,
        )
    )
    if not matches:
        return None
    return matches[-1].group(1)
