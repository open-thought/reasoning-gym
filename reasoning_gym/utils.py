import math
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any, Optional, Union

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


def format_number(num: Union[int, float], max_decimals: int = 2) -> str:
    """Convert a number to string representation with controlled decimal places.

    Args:
        num: Number to format
        max_decimals: Maximum allowed decimal places

    Returns:
        String representation of the number

    Raises:
        ValueError: If number requires more decimal places than allowed
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
        if required_decimals > max_decimals:
            raise ValueError(f"Number {num} requires {required_decimals} decimals but only {max_decimals} allowed")

    # Format with required decimals
    result = f"{num:.{max_decimals}f}".rstrip("0").rstrip(".")

    # Verify result parses back to original value
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
