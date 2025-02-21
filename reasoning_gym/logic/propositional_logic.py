"""Propositional logic task generator"""

import re
from dataclasses import dataclass
from enum import StrEnum
from random import Random
from typing import Any, Optional

from ..factory import ProceduralDataset, register_dataset


def parse_expr(expr: str):
    expr = expr.strip()
    if not expr:
        raise ValueError("Empty expression")

    if expr[0] == "(" and expr[-1] == ")":
        level = 0
        valid_enclosure = True
        for char in expr[1:-1]:
            if char == "(":
                level += 1
            elif char == ")":
                level -= 1
                if level < 0:
                    valid_enclosure = False
                    break
        if level == 0 and valid_enclosure:
            return parse_expr(expr[1:-1])

    operators_by_precedence = [[Operator.IFF], [Operator.IMPLIES], [Operator.OR], [Operator.AND]]

    for operator_level in operators_by_precedence:
        level = 0
        for i in range(len(expr) - 1, -1, -1):
            char = expr[i]
            if char == ")":
                level += 1
            elif char == "(":
                level -= 1
            elif level == 0:
                for operator in operator_level:
                    if expr[i : i + len(operator.value)] == operator.value:
                        left_expr = expr[:i]
                        right_expr = expr[i + len(operator.value) :]
                        return Expression(operator, parse_expr(left_expr), parse_expr(right_expr))

    if expr.startswith(Operator.NOT.value):
        sub_expr = expr[len(Operator.NOT.value) :]
        return Expression(Operator.NOT, parse_expr(sub_expr))

    return Expression(None, expr)


class Operator(StrEnum):
    """Basic logical operators"""

    AND = "∧"
    OR = "∨"
    NOT = "¬"
    IMPLIES = "→"
    IFF = "↔"


QUESTION_FORMAT = "\n".join(
    [
        "The following question is a propositional logic reasoning question.",
        "In the question we provide a list of premises",
        "The task is to infer a correct conclusion from the premise.",
        "FORMAT INSTRUCTIONS:",
        "Return the conclusion logic statement, as your final answer.",
        "Use the following notation to denote symbols",
        "OR = \u2228",
        "AND = \u2227",
        "IMPLIES = \u2192",
        "IFF = \u2194",
        "NOT = \u00ac",
        "Here is the question:",
    ]
)


@dataclass
class PropositionalLogicConfig:
    """Configuration for propositional logic task generation"""

    min_vars: int = 2  # Minimum number of variables
    max_vars: int = 4  # Maximum number of variables
    min_statements: int = 2  # Minimum number of given statements
    max_statements: int = 4  # Maximum number of statements
    max_complexity: int = 3  # Maximum operator depth
    seed: Optional[int] = None
    size: int = 500  # Virtual dataset size

    def validate(self):
        """Validate configuration parameters"""
        assert self.min_vars > 0, "min_vars must be positive"
        assert self.max_vars >= self.min_vars, "max_vars must be >= min_vars"
        assert self.min_statements > 0, "min_statements must be positive"
        assert self.max_statements >= self.min_statements
        assert self.max_complexity > 0, "max_complexity must be positive"


class Expression:
    """Represents a logical expression that can be evaluated"""

    def __init__(self, operator: Optional[Operator], left: Any, right: Optional[Any] = None):
        self.operator = operator
        self.left = left
        self.right = right

    def evaluate(self, assignments: dict[str, bool]) -> bool:
        """Evaluate expression with given variable assignments"""
        if self.operator is None:
            return assignments[self.left]  # Variable
        elif self.operator == Operator.NOT:
            return not self.left.evaluate(assignments)
        elif self.operator == Operator.AND:
            return self.left.evaluate(assignments) and self.right.evaluate(assignments)
        elif self.operator == Operator.OR:
            return self.left.evaluate(assignments) or self.right.evaluate(assignments)
        elif self.operator == Operator.IMPLIES:
            return (not self.left.evaluate(assignments)) or self.right.evaluate(assignments)
        elif self.operator == Operator.IFF:
            return self.left.evaluate(assignments) == self.right.evaluate(assignments)
        raise ValueError(f"Unknown operator: {self.operator}")

    @classmethod
    def from_string(cls, expr: str) -> "Expression":
        parsed_expr = parse_expr(expr)
        return cls(parsed_expr.operator, parsed_expr.left, parsed_expr.right)

    def simplify(self):
        if self.operator is None:
            return self

        simplified_left = self.left.simplify() if isinstance(self.left, Expression) else self.left
        simplified_right = self.right.simplify() if self.right and isinstance(self.right, Expression) else self.right

        if self.operator == Operator.NOT:
            if isinstance(simplified_left, Expression) and simplified_left.operator == Operator.NOT:
                return simplified_left.left
            return Expression(Operator.NOT, simplified_left)

        if self.operator in {Operator.AND, Operator.OR}:
            if simplified_left is False and self.operator == Operator.OR:
                return simplified_right
            if simplified_left is True and self.operator == Operator.AND:
                return simplified_right

            if (simplified_left is True and self.operator == Operator.OR) or (
                simplified_left is False and self.operator == Operator.AND
            ):
                return simplified_left

            if simplified_left == simplified_right:
                return simplified_left

        if self.operator == Operator.IMPLIES:
            return Expression(Operator.OR, Expression(Operator.NOT, simplified_left), simplified_right).simplify()

        return Expression(self.operator, simplified_left, simplified_right)

    def __str__(self) -> str:
        if self.operator is None:
            return self.left
        elif self.operator == Operator.NOT:
            return f"{self.operator.value}{self.left}"
        else:
            return f"({self.left} {self.operator.value} {self.right})"


class PropositionalLogicDataset(ProceduralDataset):
    """Generates propositional logic reasoning tasks"""

    def __init__(self, config: PropositionalLogicConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def __len__(self) -> int:
        return self.config.size

    def __iter__(self):
        self._current_idx = 0
        return self

    def __next__(self):
        if self._current_idx >= self.config.size:
            raise StopIteration
        item = self[self._current_idx]
        self._current_idx += 1
        return item

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Generate a single propositional logic task"""
        rng = Random(self.seed + idx)

        # Generate random variables
        num_vars = rng.randint(self.config.min_vars, self.config.max_vars)
        variables = [chr(ord("P") + i) for i in range(num_vars)]

        # Generate premises
        num_statements = rng.randint(self.config.min_statements, self.config.max_statements)
        premises = self._generate_premises(rng, variables, num_statements)
        conclusion = self._find_valid_conclusion(rng, premises, variables)

        # Format question
        question = QUESTION_FORMAT
        question += "Given:\n"
        for i, premise in enumerate(premises, 1):
            question += f"{i}. {premise}\n."
        question += "What can we conclude from the above statements?"

        return {
            "question": question,
            "answer": None,
            "metadata": {
                "premises": [str(p) for p in premises],
                "variables": variables,
                "complexity": self._measure_complexity(conclusion),
                "example_answer": str(conclusion),
            },
        }

    def _generate_premises(self, rng: Random, variables: list[str], num_statements: int) -> list[Expression]:
        """Generate a list of premise statements"""
        premises = []
        for _ in range(num_statements):
            depth = rng.randint(1, self.config.max_complexity)
            premises.append(self._generate_expression(rng, variables, depth))
        return premises

    def _generate_expression(self, rng: Random, variables: list[str], depth: int) -> Expression:
        """Generate a random logical expression"""
        if depth <= 1:
            return Expression(None, rng.choice(variables))
        operator = rng.choice(list(Operator))
        if operator == Operator.NOT:
            return Expression(operator, self._generate_expression(rng, variables, depth - 1))
        else:
            left = self._generate_expression(rng, variables, depth - 1)
            right = self._generate_expression(rng, variables, depth - 1)
            return Expression(operator, left, right)

    def _find_valid_conclusion(self, rng: Random, premises: list[Expression], variables: list[str]) -> Expression:
        """Find a valid conclusion that follows from the premises"""
        for _ in range(100):
            candidate = self._generate_expression(rng, variables, 2).simplify()
            if self._is_valid_conclusion(premises, candidate) and not (self._is_trivial(candidate)):
                return candidate

        # Fallback to a simple conclusion
        return Expression(None, variables[0])

    def _is_valid_conclusion(self, premises: list[Expression], conclusion: Expression) -> bool:
        """Check if conclusion follows from premises using truth tables"""
        variables = self._collect_variables(premises + [conclusion])

        # Check all possible assignments
        for assignment in self._generate_assignments(variables):
            # If premises are true but conclusion is false, invalid
            if all(p.evaluate(assignment) for p in premises) and not conclusion.evaluate(assignment):
                return False
        return True

    def _collect_variables(self, expressions: list[Expression]) -> set[str]:
        """Collect all variables used in expressions"""
        variables = set()
        for expr in expressions:
            if expr.operator is None:
                variables.add(expr.left)
            else:
                if isinstance(expr.left, Expression):
                    variables.update(self._collect_variables([expr.left]))
                if expr.right and isinstance(expr.right, Expression):
                    variables.update(self._collect_variables([expr.right]))
        return variables

    def _generate_assignments(self, variables: set[str]) -> list[dict[str, bool]]:
        """Generate all possible truth value assignments"""
        assignments = []
        for i in range(2 ** len(variables)):
            assignment = {}
            for j, var in enumerate(sorted(variables)):
                assignment[var] = bool((i >> j) & 1)
            assignments.append(assignment)
        return assignments

    def _measure_complexity(self, expression: Expression) -> int:
        """Measure the complexity of an expression"""
        if expression.operator is None:
            return 1
        elif expression.operator == Operator.NOT:
            return 1 + self._measure_complexity(expression.left)
        else:
            return 1 + self._measure_complexity(expression.left) + self._measure_complexity(expression.right)

    def score_answer(self, answer: str | None, entry: dict[str, Any]) -> float:
        """Robust scoring implementation for propositional logic answers"""
        if not answer:
            return 0.0

        try:
            cleaned_answer = answer

            valid_vars = set(entry["metadata"]["variables"])
            answer_vars = re.findall(r"([A-Z])", cleaned_answer)
            if any(var not in valid_vars for var in answer_vars):
                return 0.01

            premises = [Expression.from_string(p) for p in entry["metadata"]["premises"]]
            answer_expr = Expression.from_string(cleaned_answer)

            if self._is_valid_conclusion(premises, answer_expr):
                if self._is_trivial(answer_expr):
                    return 0.25
                else:
                    return 1.0
            return 0.05
        except (ValueError, KeyError, AttributeError):
            return 0.01

    def _is_trivial(self, expr: Expression) -> bool:
        """Check for trivial tautologies like P ∨ ¬P"""
        if expr.operator is None:
            return True
        variables = self._collect_variables([expr])
        for assignment in self._generate_assignments(variables):
            if not expr.evaluate(assignment):
                return False
        return True


register_dataset("propositional_logic", PropositionalLogicDataset, PropositionalLogicConfig)
