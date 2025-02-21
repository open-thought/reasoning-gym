import ast
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, getcontext
from random import Random
from typing import Any, Optional

from ..factory import ProceduralDataset, register_dataset


@dataclass
class DecimalArithmeticConfig:
    """Configuration for decimal arithmetic dataset generation"""

    min_num_decimal_places: int = 6
    max_num_decimal_places: int = 6
    precision: int = 28
    terms: int = 6
    seed: Optional[int] = None
    size: int = 500

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert (
            self.precision > self.max_num_decimal_places + 1
        ), "precision must be 2 or more higher than max_num_decimal_places"


def build_grouped_expression(operands: list[str], operators: list[str], rng: Random) -> str:
    """
    Recursively build an arithmetic expression string from operands and operators,
    inserting parentheses at random.

    The expression is built by choosing a random split among the operands;
    the operator at that split becomes the “root” of the subexpression.
    With 50% chance, the resulting combination is wrapped in parentheses.
    """
    if len(operands) == 1:
        return operands[0]
    # Randomly choose a split point (1 <= split < len(operands)).
    split: int = rng.randint(1, len(operands) - 1)
    left_expr: str = build_grouped_expression(operands[:split], operators[: split - 1], rng)
    right_expr: str = build_grouped_expression(operands[split:], operators[split:], rng)
    # The operator at position (split - 1) is the one combining the two groups.
    expr: str = left_expr + operators[split - 1] + right_expr
    # Randomly decide to add parentheses around this subexpression.
    if rng.choice([True, False]):
        expr = "(" + expr + ")"
    return expr


def generate_arithmetic_problem(
    rng: Random,
    min_num_decimal_places: int,
    max_num_decimal_places: int,
    terms: int = 2,
    operations: Optional[list[str]] = None,
) -> str:
    """
    Generates a simple arithmetic problem with decimal numbers (as a string) formatted
    to a specific number of decimal places, with random parenthesis grouping.

    Parameters:
        rng: Random number generator.
        min_num_decimal_places (int): Minimum number of decimal places.
        max_num_decimal_places (int): Maximum number of decimal places.
        terms (int): Number of numbers in the arithmetic expression.
        operations (list): List of operations to use (default: ['+', '-', '*', '/']).

    Returns:
        str: A formatted arithmetic expression ending with " = ?"
    """
    if operations is None:
        operations = ["+", "-", "*", "/"]

    operands: list[str] = []
    operators: list[str] = []

    for i in range(terms):
        # Choose a random number of decimal places for this term.
        ndp: int = rng.randint(min_num_decimal_places, max_num_decimal_places)
        max_integer_part: int = 10  # Maximum whole number before the decimal
        max_value: int = max_integer_part * (10**ndp)
        raw_int: int = rng.randint(1, max_value)
        # Create the Decimal number and quantize it to exactly ndp decimal places.
        num: Decimal = Decimal(raw_int) / (Decimal(10) ** ndp)
        quantize_str: str = "1." + "0" * ndp
        num = num.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)
        # Format the number as a string with exactly ndp decimals.
        num_str: str = f"{num:.{ndp}f}"
        operands.append(num_str)
        if i < terms - 1:
            op: str = rng.choice(operations)
            operators.append(op)

    expr: str = build_grouped_expression(operands, operators, rng)
    problem_str: str = expr + " = ?"
    return problem_str


def evaluate_expression(expr: str) -> Decimal:
    """
    Safely evaluates a simple arithmetic expression using AST parsing, performing
    all arithmetic in the Decimal context.

    Args:
        expr: A string containing the arithmetic expression.

    Returns:
        Decimal: The computed result.
    """
    tree: ast.Expression = ast.parse(expr, mode="eval")
    return _eval_ast(tree.body)


def _eval_ast(node: ast.AST) -> Decimal:
    """Recursively evaluate an AST node using Decimal arithmetic."""
    if isinstance(node, ast.BinOp):
        left: Decimal = _eval_ast(node.left)
        right: Decimal = _eval_ast(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        elif isinstance(node.op, ast.Sub):
            return left - right
        elif isinstance(node.op, ast.Mult):
            return left * right
        elif isinstance(node.op, ast.Div):
            return left / right
        else:
            raise ValueError(f"Unsupported operator: {node.op}")
    elif isinstance(node, ast.UnaryOp):
        operand: Decimal = _eval_ast(node.operand)
        if isinstance(node.op, ast.UAdd):
            return operand
        elif isinstance(node.op, ast.USub):
            return -operand
        else:
            raise ValueError(f"Unsupported unary operator: {node.op}")
    elif isinstance(node, ast.Constant):  # For Python 3.8+
        return Decimal(str(node.value))
    elif isinstance(node, ast.Num):  # For older Python versions
        return Decimal(str(node.n))
    else:
        raise ValueError(f"Unsupported expression component: {node}")


class DecimalArithmeticDataset(ProceduralDataset):
    """Dataset that generates basic arithmetic tasks using Decimal arithmetic and proper operator precedence."""

    def __init__(self, config: DecimalArithmeticConfig) -> None:
        super().__init__(config=config, seed=config.seed, size=config.size)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Generate a single arithmetic task.

        Returns:
            dict: Contains:
              - 'question': The formatted arithmetic expression as a string.
              - 'answer': The computed Decimal result.
              - 'metadata': Additional metadata (currently empty).
        """
        # Create a deterministic RNG from base seed and index.
        rng: Random = Random(self.seed + idx if self.seed is not None else None)
        getcontext().prec = self.config.precision

        problem_str: str = generate_arithmetic_problem(
            rng,
            self.config.min_num_decimal_places,
            self.config.max_num_decimal_places,
            terms=self.config.terms,
        )
        # Remove the trailing " = ?" to obtain the pure arithmetic expression.
        expr: str = problem_str.replace(" = ?", "").strip()
        answer: Decimal = evaluate_expression(expr)

        problem_str = (
            f"Please solve this problem to a maximum of {str(self.config.precision)} significant digits, rounding up from the half. Only reply with the final value.\n"
            + problem_str
        )

        return {"question": problem_str, "answer": answer, "metadata": {}}

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """
        Compares the user's answer (converted to Decimal) with the correct answer.
        Instead of requiring exact equality, we allow an error up to one unit in the
        least significant digit as determined by the level of precision (max_num_decimal_places).

        Returns:
            float: 1.0 if the user's answer is within tolerance; otherwise, 0.01.
        """
        if answer is None:
            return 0.0

        try:
            user_ans: Decimal = Decimal(answer)
            correct_ans: Decimal = entry["answer"]

            # Determine tolerance based on the desired precision.
            precision: int = self.config.max_num_decimal_places
            tol: Decimal = Decimal(10) ** (-precision)
            if abs(user_ans - correct_ans) <= tol:
                return 1.0
        except Exception:
            return 0.01

        return 0.01


# Register the dataset with the factory.
register_dataset("decimal_arithmetic", DecimalArithmeticDataset, DecimalArithmeticConfig)
