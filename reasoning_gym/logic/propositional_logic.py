"""Propositional logic task generator"""

from dataclasses import dataclass
from enum import StrEnum
from random import Random
from typing import Any, List, Optional, Set

from ..factory import ProceduralDataset, register_dataset


class Operator(StrEnum):
    """Basic logical operators"""

    AND = "∧"
    OR = "∨"
    NOT = "¬"
    IMPLIES = "→"
    IFF = "↔"


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

        # Generate a valid conclusion
        conclusion = self._find_valid_conclusion(rng, premises, variables)

        # Format question
        question = "Given:\n"
        for i, premise in enumerate(premises, 1):
            question += f"{i}. {premise}\n"
        question += "What can we conclude?"

        return {
            "question": question,
            "answer": str(conclusion),
            "metadata": {
                "premises": [str(p) for p in premises],
                "variables": variables,
                "complexity": self._measure_complexity(conclusion),
            },
        }

    def _generate_premises(self, rng: Random, variables: List[str], num_statements: int) -> List[Expression]:
        """Generate a list of premise statements"""
        premises = []
        for _ in range(num_statements):
            depth = rng.randint(1, self.config.max_complexity)
            premises.append(self._generate_expression(rng, variables, depth))
        return premises

    def _generate_expression(self, rng: Random, variables: List[str], depth: int) -> Expression:
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

    def _find_valid_conclusion(self, rng: Random, premises: List[Expression], variables: List[str]) -> Expression:
        """Find a valid conclusion that follows from the premises"""
        # Try random conclusions until we find a valid one
        for _ in range(100):
            candidate = self._generate_expression(rng, variables, 2)
            if self._is_valid_conclusion(premises, candidate):
                return candidate

        # Fallback to a simple conclusion
        return Expression(None, variables[0])

    def _is_valid_conclusion(self, premises: List[Expression], conclusion: Expression) -> bool:
        """Check if conclusion follows from premises using truth tables"""
        variables = self._collect_variables(premises + [conclusion])

        # Check all possible assignments
        for assignment in self._generate_assignments(variables):
            # If premises are true but conclusion is false, invalid
            if all(p.evaluate(assignment) for p in premises) and not conclusion.evaluate(assignment):
                return False
        return True

    def _collect_variables(self, expressions: List[Expression]) -> Set[str]:
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

    def _generate_assignments(self, variables: Set[str]) -> List[dict[str, bool]]:
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


register_dataset("propositional_logic", PropositionalLogicDataset, PropositionalLogicConfig)
