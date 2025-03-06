import math
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any, Optional, Union

SYSTEM_PROMPTS = {
    "DeepSeekZero": """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer>answer here</answer>
Do not explain your reasoning inside the answer tags, provide only the final answer. When an example is provided, you should strictly follow the format of the output/answer in that example.
""",
    "default": """Given a problem, your task is to answer the question by thinking step-by-step in a clear and specific manner.
Once you have thought about the reasoning process, provide the answer in the following format:
<answer>answer here</answer>
Do not explain your reasoning inside the answer tags, provide only the final answer. When an example is provided, you should strictly follow the format of the output/answer in that example.
""",
    "simple": "You are a helpful assistant that answers questions accurately and concisely. When asked to solve a problem, show your work step by step. Provide your final answer between <answer> and </answer> tags.",
    "direct": "Answer the question directly. Provide your answer between <answer> and </answer> tags. Do not return any preamble, explanation, or reasoning.",
    "chain_of_draft": "Think step by step, but only keep a minimum draft for each thinking step, with 5 words at most. Return the answer at the end of the response enclosed in <answer> </answer> tags.",
}


def extract_answer(completion: str, tag_name: str = "answer", strip: bool = True) -> Optional[str]:
    regex = f"<{tag_name}>\\s?(.*?)\\s?</{tag_name}>"
    matches = list(
        re.finditer(
            regex,
            completion,
            flags=re.DOTALL,
        )
    )
    if not matches:
        return None
    answer = matches[-1].group(1)
    if strip:
        answer = answer.strip()
    return answer


def format_number(num: Union[int, float], max_decimals: int = 2, round_if_needed: bool = False) -> str:
    """Convert a number to string representation with controlled decimal places.

    Args:
        num: Number to format
        max_decimals: Maximum allowed decimal places
        round_if_needed: If True, round the number to max_decimals instead of raising an error

    Returns:
        String representation of the number

    Raises:
        ValueError: If number requires more decimal places than allowed and round_if_needed is False
    """
    if isinstance(num, int) or num.is_integer():
        return str(int(num))

    # Convert to Decimal for exact decimal arithmetic
    d = Decimal(str(num))

    # Find required decimals by removing trailing zeros
    str_val = f"{d:f}"
    str_val = str_val.rstrip("0").rstrip(".")
    if "." in str_val:
        required_decimals = len(str_val.split(".")[1])
        if required_decimals > max_decimals and not round_if_needed:
            raise ValueError(f"Number {num} requires {required_decimals} decimals but only {max_decimals} allowed")

    # Format with required decimals (will round if needed)
    result = f"{num:.{max_decimals}f}".rstrip("0").rstrip(".")

    # Verify result parses back to original value (skip verification if rounding was applied)
    if not round_if_needed:
        try:
            parsed = float(result)
            if not math.isclose(parsed, num, rel_tol=1e-9):
                raise ValueError(f"String representation {result} does not match original value {num}")
        except (ValueError, InvalidOperation) as e:
            raise ValueError(f"Failed to verify string representation: {e}")

    return result


def is_integer(obj: Any) -> bool:
    if isinstance(obj, (int, float)):
        return isinstance(obj, int) or obj.is_integer()
    elif isinstance(obj, Fraction):
        return obj.denominator == 1
    return False


def compute_decimal_reward(answer: Optional[str], oracle_answer: str, strip_commas: bool = True) -> float:
    """Compute the reward for a given answer compared to the oracle answer.
    Tolerate sign, leading zeros and trailing decimals, optionally strip commas ("+01,000.00" == "1000")

    Args:
        answer: Answer provided by model
        oracle_answer: Correct answer to the question
        strip_commas: Whether to remove commas from answers e.g "1,000" = "1000"

    Returns:
        Reward value between 0.0 and 1.0
    """
    reward = 0.0
    if answer is not None and len(answer) > 0:
        try:
            if strip_commas:
                answer = answer.replace(",", "")
                oracle_answer = oracle_answer.replace(",", "")

            if Decimal(answer) == Decimal(oracle_answer):
                return 1.0
        except:
            pass

        if oracle_answer in answer:
            reward = len(oracle_answer) / len(answer)

    return reward
