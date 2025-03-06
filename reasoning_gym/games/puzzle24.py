import re
from dataclasses import dataclass
from random import Random
from typing import Any, Optional

import sympy
from sympy import Symbol, symbols
from sympy.parsing.sympy_parser import parse_expr

from ..factory import ProceduralDataset, register_dataset

QUESTION_TEMPLATE = """Make 24 using {numbers}. You can only use each number once. You can use the operators {operators}.
Final answer format instructions:
1. Provide your final answer as a arithmetic expression (no '=' sign).
2. Do not include the target number in the expression.
3. Use '*' for multiplication.
4. Use '/' for division.
"""


@dataclass
class Puzzle24Config:
    operators: tuple = ("+", "-", "*", "/")
    min_value: int = 1
    max_value: int = 10
    seed: Optional[int] = None
    size: int = 500

    def validate(self):
        assert len(self.operators) > 0, "At least one operator is required"
        assert self.min_value <= self.max_value, "Minimum value must be less than or equal to maximum value"
        assert self.min_value >= 1, "Minimum value must be at least 1"
        assert self.max_value <= 10, "Maximum value must be at most 10"
        assert self.size > 0, "Size must be greater than 0"


class Puzzle24Dataset(ProceduralDataset):
    def __init__(self, config: Puzzle24Config):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _generate_candidate_expression(self, rng: Random, num_terms: int) -> tuple[sympy.Expr, list[int], list[Symbol]]:
        """Generate a candidate expression with random numbers and operators

        Args:
            rng: Random number generator
            num_terms: Number of terms to include

        Returns:
            Tuple of (sympy expression, list of numbers, list of symbols)
        """
        # Generate random numbers
        numbers = [rng.randint(self.config.min_value, self.config.max_value) for _ in range(num_terms)]

        # Create symbols for building expression
        syms = symbols(f"x:{num_terms}")

        # Build random expression
        expr = syms[0]

        for i in range(1, num_terms):
            op = rng.choice(self.config.operators)
            if op == "+":
                expr = expr + syms[i]
            elif op == "-":
                expr = expr - syms[i]
            elif op == "*":
                expr = expr * syms[i]
            else:
                if numbers[i] != 0:
                    current = int(expr.subs({sym: num for sym, num in zip(syms[:i], numbers[:i])}))
                    remaining = [n for n in numbers[i:] if n != 0]
                    rng.shuffle(remaining)
                    found_divisor = False
                    for div in remaining:
                        if current % div == 0:
                            numbers[i] = div
                            expr = expr / syms[i]
                            found_divisor = True
                            break
                    if not found_divisor:
                        expr = expr - syms[i]
                else:
                    expr = expr + syms[i]

        return expr, numbers, syms

    def __getitem__(self, idx: int) -> dict:
        rng = Random(self.seed + idx)
        while True:
            expr, numbers, syms = self._generate_candidate_expression(rng, 4)
            if expr.subs({sym: num for sym, num in zip(syms, numbers)}) == 24:
                break
        expr_str = str(expr)
        for i, sym in enumerate(syms):
            expr_str = expr_str.replace(str(sym), str(numbers[i]))

        question = QUESTION_TEMPLATE.format(
            numbers=", ".join(map(str, numbers)), operators=", ".join(self.config.operators)
        )
        return {
            "question": question,
            "answer": expr_str,
            "metadata": {
                "numbers": numbers,
                "expression": expr,
            },
        }

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        reward = 0.01
        if answer is not None:
            try:
                answer = answer.strip()
                user_answer = int(parse_expr(answer))
                solved = user_answer == 24
                used_numbers = [int(num) for num in re.findall(r"\b\d+\b", answer)]
                if len(used_numbers) != 4:
                    reward = 0.01
                elif any(num > self.config.max_value or num < self.config.min_value for num in used_numbers):
                    reward = 0.01
                elif not solved:
                    reward = 0.01
                else:
                    reward = 1.0
            except Exception as e:
                reward = 0.01
        return reward


register_dataset("puzzle24", Puzzle24Dataset, Puzzle24Config)
