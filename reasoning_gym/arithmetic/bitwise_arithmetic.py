from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from ..factory import ProceduralDataset, register_dataset


@dataclass
class BitwiseArithmeticConfig:
    """Configuration for Bitwise arithmetic dataset generation"""

    difficulty: int = 2  # Controls expression complexity: 1=simple expressions, 2=nested expressions, 3+=deeper nesting
    seed: Optional[int] = None
    size: int = 500

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert 0 < self.difficulty, "difficulty must be gt 0"
        assert 10 >= self.difficulty, "difficulty must be lte 10"


def generate_expression(rng: Random, max_depth: int) -> str:
    """
    Recursively generate a random arithmetic expression that includes
    standard arithmetic (+, -, *) and bitwise shifting (<<, >>) operators.
    All numbers are represented in hexadecimal format as multi-byte values.

    Parameters:
        rng (Random): Random number generator instance
        max_depth (int): Maximum depth of nested expressions.

    Returns:
        str: A string representing the generated expression.
    """
    # Base case: return a random multi-byte number in hex (0x100 to 0xFFFF).
    if max_depth <= 0:
        return hex(rng.randint(0x100, 0xFFFF))

    # Occasionally return a simple hex number even if max_depth > 0.
    if rng.random() < 0.01:
        return hex(rng.randint(0x100, 0xFFFF))

    # Choose a random operator.
    operators = ["+", "-", "*", "<<", ">>"]
    op = rng.choice(operators)

    # Generate left and right subexpressions.
    left_expr = generate_expression(rng, max_depth - 1)
    right_expr = generate_expression(rng, max_depth - 1)

    # For bitwise shift operations, keep the right operand small (in hex).
    if op in ["<<", ">>"]:
        right_expr = hex(rng.randint(0, 3))

    return f"({left_expr} {op} {right_expr})"


def generate_problem(rng: Random, difficulty: int = 1) -> tuple[str, str]:
    """
    Generate a random arithmetic problem involving multi-byte hexadecimal numbers.

    The 'difficulty' parameter controls the complexity:
      - difficulty=1: Simple expressions like (0x123 + 0x456)
      - difficulty=2: Nested expressions like ((0x123 + 0x456) << 1)
      - difficulty=3: More complex nesting like ((0x123 + 0x456) << (0x789 >> 1))
      Higher values continue to increase nesting depth and expression complexity.

    Parameters:
        rng (Random): Random number generator instance
        difficulty (int): The difficulty level (1 = simplest; higher values = more complex).

    Returns:
        tuple: (problem_str, correct_answer)
          - problem_str (str): The generated arithmetic expression (with hex numbers).
          - correct_answer (str): The evaluated result, formatted as a hex string.
    """
    max_depth = max(1, difficulty)
    problem_str = generate_expression(rng, max_depth)
    correct_value = eval(problem_str)
    correct_answer = hex(correct_value)

    return problem_str, correct_answer


def verify_solution(problem, user_solution):
    """
    Verify if the provided solution is correct for the given problem.

    Parameters:
        problem (str): The arithmetic expression (with hex numbers).
        user_solution (str or int): The user's answer, either as a hex string (e.g., "0xa")
            or an integer.

    Returns:
        bool: True if the user's answer matches the evaluated result, else False.
    """
    try:
        correct_value = eval(problem)
        # Use base=0 for automatic base detection: 0x->hex, 0b->binary, 0o->octal, no prefix->decimal
        user_value = int(str(user_solution), 0)
    except Exception:
        return False

    return correct_value == user_value


class BitwiseArithmeticDataset(ProceduralDataset):
    """Dataset that generates tasks testing understanding of bitwise arithmetic operations.

    Generates expressions combining:
    - Standard arithmetic operators (+, -, *)
    - Bitwise shift operators (<<, >>)
    - Multi-byte hexadecimal numbers (e.g. 0x100 to 0xFFFF)

    The difficulty parameter controls expression complexity:
    - Level 1: Simple expressions like (0x123 + 0x456)
    - Level 2: Nested expressions with shifts like ((0x123 + 0x456) << 1)
    - Level 3+: Deeper nesting like ((0x123 + 0x456) << (0x789 >> 1))

    Each task provides:
    - A question asking to evaluate an expression
    - The correct answer in hexadecimal format
    - Metadata including the raw expression

    The dataset verifies answers by evaluating them as Python expressions,
    supporting both integer and hexadecimal string formats.
    """

    def __init__(self, config: BitwiseArithmeticConfig) -> None:
        super().__init__(config=config, seed=config.seed, size=config.size)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Generate a single arithmetic task.

        Returns:
            dict: Contains:
              - 'question': The formatted arithmetic expression as a string.
              - 'answer': The computed hexidecimal result.
              - 'metadata': Additional metadata, including just the problem without prompt.
        """
        # Create a deterministic RNG from base seed and index.
        rng: Random = Random(self.seed + idx if self.seed is not None else None)

        problem, answer = generate_problem(
            rng,
            self.config.difficulty,
        )
        problem_str = (
            f"Please solve this problem. Assume there is arbitrary bit depth and that there are signed integers. Reply only with the final hexidecimal value.\n"
            + problem
        )

        return {"question": problem_str, "answer": answer, "metadata": {"problem": problem}}

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """
        Compares the user's answer with the correct answer.

        Returns:
            float: 1.0 if the user's answer is correct; otherwise, 0.01 unless no answer is provided, in which case 0.
        """
        if answer is None:
            return 0.0

        try:
            solved = verify_solution(entry["metadata"]["problem"], answer)
            if solved:
                return 1.0
        except Exception:
            return 0.01

        return 0.01


# Register the dataset with the factory.
register_dataset("bitwise_arithmetic", BitwiseArithmeticDataset, BitwiseArithmeticConfig)
