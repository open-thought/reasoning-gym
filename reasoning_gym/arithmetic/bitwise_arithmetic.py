import ast
from dataclasses import dataclass
from random import Random
from typing import Any, Dict, List, Optional

from ..factory import ProceduralDataset, register_dataset


@dataclass
class BitwiseArithmeticConfig:
    """Configuration for Bitwise arithmetic dataset generation"""

    difficulty: int = 2
    seed: Optional[int] = None
    size: int = 500

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert 0 < self.difficulty, "difficulty must be gt 0"
        assert 10 >= self.difficulty, "difficulty must be lte 10"


def generate_expression(rng, max_depth):
    """
    Recursively generate a random arithmetic expression that includes
    standard arithmetic (+, -, *) and bitwise shifting (<<, >>) operators.
    All numbers are represented in hexadecimal format as multi-byte values.

    Parameters:
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


def generate_problem(rng, difficulty=1):
    """
    Generate a random arithmetic problem involving multi-byte hexadecimal numbers.

    The 'difficulty' parameter controls the complexity:
      - Lower difficulty produces a shallower expression.
      - Higher difficulty produces a more deeply nested expression.

    Parameters:
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
        user_value = int(str(user_solution), 0)
    except Exception as e:
        return False

    return correct_value == user_value


class BitwiseArithmeticDataset(ProceduralDataset):
    """Dataset that generates basic tasks using bitwise arithmetic and proper operator precedence."""

    def __init__(self, config: BitwiseArithmeticConfig) -> None:
        super().__init__(config=config, seed=config.seed, size=config.size)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Generate a single arithmetic task.

        Returns:
            dict: Contains:
              - 'question': The formatted arithmetic expression as a string.
              - 'answer': The computed hexidecimal result.
              - 'metadata': Additional metadata.
        """
        # Create a deterministic RNG from base seed and index.
        rng: Random = Random(self.seed + idx if self.seed is not None else None)

        problem, answer = generate_problem(
            rng,
            self.config.difficulty,
        )
        problem_str = f"Please solve this problem. Reply only with the final hexidecimal value.\n" + problem

        return {"question": problem_str, "answer": answer, "metadata": {"problem": problem}}

    def score_answer(self, answer: Optional[str], entry: Dict[str, Any]) -> float:
        """
        Compares the user's answer (converted to Bitwise) with the correct answer.
        Instead of requiring exact equality, we allow an error up to one unit in the
        least significant digit as determined by the level of precision (max_num_Bitwise_places).

        Returns:
            float: 1.0 if the user's answer is within tolerance; otherwise, 0.01.
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
register_dataset("Bitwise_arithmetic", BitwiseArithmeticDataset, BitwiseArithmeticConfig)
