import random
import string
from dataclasses import dataclass
from typing import Optional, Tuple

from sympy import Symbol

from ..factory import ProceduralDataset, register_dataset


@dataclass
class SimpleEquationsConfig:
    """Configuration for simple equation task generation"""

    min_terms: int = 2  # Minimum number of terms in expression
    max_terms: int = 4  # Maximum number of terms
    min_value: int = 1  # Minimum value for constants
    max_value: int = 100  # Maximum value for constants
    operators: tuple = ("+", "-", "*")  # Allowed operators
    seed: Optional[int] = None
    size: int = 500

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.min_terms > 0, "min_terms must be positive"
        assert self.max_terms >= self.min_terms, "max_terms must be >= min_terms"
        assert self.min_value > 0, "min_value must be positive"
        assert self.max_value >= self.min_value, "max_value must be >= min_value"
        assert len(self.operators) > 0, "must specify at least one operator"
        assert all(op in ("+", "-", "*") for op in self.operators), "invalid operator specified"


class SimpleEquationsDataset(ProceduralDataset):
    """Generates simple equations with one variable to solve"""

    def __init__(self, config: SimpleEquationsConfig):
        self._prompt_templates = [
            "Find the value of {variable} in the equation: {equation}",
            "Solve for {variable}: {equation}",
            "Determine the value of {variable} that satisfies: {equation}",
        ]
        super().__init__(config=config, seed=config.seed, size=config.size)

    def __getitem__(self, idx: int) -> dict:
        """Generate a single equation task

        Returns:
            dict with keys:
                - question: str, the equation to solve (e.g. "3 * x = 12")
                - answer: str, the solution value (e.g. "4")
                - metadata: dict with generation parameters
        """
        rng = random.Random(self.seed + idx)

        # Get variable and generate equation
        variable = self._get_variable(rng)
        equation, solution = self._generate_equation(rng, variable)

        return {
            "question": rng.choice(self._prompt_templates).format(variable=variable, equation=equation),
            "answer": str(solution),
            "metadata": {
                "equation": equation,
                "variable": variable,
            },
        }

    def _get_variable(self, rng: random.Random) -> str:
        """Get a random lowercase variable name"""
        return rng.choice(string.ascii_lowercase)

    def _generate_equation(self, rng: random.Random, variable: str) -> Tuple[str, int]:
        """Generate an equation and its solution

        Args:
            rng: Random number generator
            variable: Variable symbol to use in equation

        Returns:
            Tuple of (equation string, solution integer)
        """
        x = Symbol(variable)

        # Generate terms for left side
        num_terms = rng.randint(self.config.min_terms, self.config.max_terms)
        terms = []

        # Generate all constant terms first
        for _ in range(num_terms):
            value = rng.randint(self.config.min_value, self.config.max_value)
            terms.append(value)

        # Replace one random term with the variable term
        var_pos = rng.randint(0, num_terms - 1)
        coef = rng.randint(self.config.min_value, self.config.max_value)
        if "*" in self.config.operators:
            terms[var_pos] = coef * x
        else:
            terms[var_pos] = x

        # Apply operators between terms
        expr = terms[0]
        for i in range(1, num_terms):
            op = rng.choice(self.config.operators)
            if op == "+":
                expr = expr + terms[i]
            elif op == "-":
                expr = expr - terms[i]
            else:  # '*'
                expr = expr * terms[i]

        left_side = expr
        solution_value = rng.randint(self.config.min_value, self.config.max_value)
        right_side = left_side.subs(x, solution_value)
        return f"{left_side} = {right_side}", solution_value


register_dataset("simple_equations", SimpleEquationsDataset, SimpleEquationsConfig)
