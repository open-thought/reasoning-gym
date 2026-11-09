import math
import re
from decimal import Decimal, InvalidOperation
from enum import Enum
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


# Match a complete decimal spelling, including properly grouped thousands.
_DECIMAL_NUMBER = re.compile(
    r"[+-]?(?:(?:0*[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
)
# Consume malformed grouping too, so it cannot expose a correct substring.
_NUMBER_CANDIDATE = re.compile(r"[+-]?(?:[0-9]+(?:,[0-9]+)*(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")


def _extract_numeric_token(answer: str) -> Optional[str]:
    """Extract one complete numeric token, allowing surrounding prose punctuation.

    This checks spelling and boundaries, not numeric equivalence. A final period
    is treated as sentence punctuation. Operators immediately around the token
    (including separated signs) cannot turn a numeric fragment into an answer.
    """
    matches = list(_NUMBER_CANDIDATE.finditer(answer))
    if len(matches) != 1:
        return None
    match = matches[0]
    token = match.group().removesuffix(".")
    before, after = answer[: match.start()], answer[match.start() + len(token) :]
    if before and (before[-1].isalnum() or before[-1] in "_.,+-−"):
        return None
    if after and (after[0].isalnum() or after[0] in "_+-"):
        return None
    # A comma or period is punctuation only at the end or before whitespace.
    if after[:1] in (".", ",") and len(after) > 1 and not after[1].isspace():
        return None
    if before.rstrip().endswith(("+", "-", "−", "/", "*", "^", "=", "%")):
        return None
    if after.lstrip().startswith(("-", "−", "/", "*", "^", "=", "%")) or after.strip() == "+":
        return None
    if re.search(r"\b(?:nan|inf|infinity)\b", before + after, re.IGNORECASE):
        return None
    return token if _DECIMAL_NUMBER.fullmatch(token) else None


def compute_string_reward(answer: Optional[str], oracle_answer: str) -> float:
    """Score literal equality or length-based substring credit.

    For numeric-looking oracles, partial credit requires exactly one complete
    numeric token with identical spelling. This also protects digit strings and
    program output without treating them as decimal values.
    """
    if not isinstance(answer, str) or not answer:
        return 0.0
    if answer == oracle_answer:
        return 1.0
    if _DECIMAL_NUMBER.fullmatch(oracle_answer):
        if _extract_numeric_token(answer) != oracle_answer:
            return 0.0
    elif oracle_answer not in answer:
        return 0.0
    return len(oracle_answer) / len(answer)


def compute_decimal_reward(answer: Optional[str], oracle_answer: str, strip_commas: bool = True) -> float:
    """Score one finite decimal answer exactly, with partial credit for prose.

    Equivalent standalone numbers receive full credit. Prose receives the
    matched token's length divided by the response length only when it contains
    exactly one complete, equivalent number. Multiple numbers, fractions,
    percentages and malformed numeric tokens receive zero. Commas, when enabled,
    must group thousands. No approximate numeric tolerance is applied.
    """
    if not isinstance(answer, str) or not answer.strip():
        return 0.0

    def parse(value: str) -> Optional[Decimal]:
        if not _DECIMAL_NUMBER.fullmatch(value) or (not strip_commas and "," in value):
            return None
        try:
            number = Decimal(value.replace(",", ""))
            return number if number.is_finite() else None
        except InvalidOperation:
            return None

    expected = parse(oracle_answer.strip())
    if expected is None:
        return 0.0
    standalone = parse(answer.strip())
    if standalone is not None:
        return float(standalone == expected)

    token = _extract_numeric_token(answer)
    if token is None or parse(token) != expected:
        return 0.0
    return len(token) / len(answer)


class StrEnum(str, Enum):
    """
    Taken from Python 3.11 StrEnum implementation, moved here to support Python 3.10.
    Enum where members are also (and must be) strings
    """

    def __new__(cls, *values):
        "values must already be of type `str`"
        if len(values) > 3:
            raise TypeError("too many arguments for str(): %r" % (values,))
        if len(values) == 1:
            # it must be a string
            if not isinstance(values[0], str):
                raise TypeError("%r is not a string" % (values[0],))
        if len(values) >= 2:
            # check that encoding argument is a string
            if not isinstance(values[1], str):
                raise TypeError("encoding must be a string, not %r" % (values[1],))
        if len(values) == 3:
            # check that errors argument is a string
            if not isinstance(values[2], str):
                raise TypeError("errors must be a string, not %r" % (values[2]))
        value = str(*values)
        member = str.__new__(cls, value)
        member._value_ = value
        return member

    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        """
        Return the lower-cased version of the member name.
        """
        return name.lower()
