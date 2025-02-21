import random
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Optional

import sympy

from ..factory import ProceduralDataset, register_dataset


@dataclass
class SimpleIntegrationConfig:
    min_terms: int = 2
    max_terms: int = 5
    min_degree: int = 1
    max_degree: int = 10
    min_bounds: int = 1
    max_bounds: int = 10
    operators: tuple = ("+", "-")
    symbols: tuple = ("x", "X")
    seed: Optional[int] = None
    size: int = 500

    def validate(self) -> None:
        """Validate the configuration parameters of the integral proble"""
        assert self.min_bounds > 0, "min_bounds must be positive"
        assert self.max_bounds >= self.min_bounds, "max_bounds must be >= min_bounds"
        assert self.min_terms >= 0, "min_terms must be positive"
        assert self.max_terms >= self.min_terms, "max_terms must be >= min_terms"
        assert self.min_degree >= -10, "min_degree must be >= -10"
        assert self.max_degree >= self.min_degree, "max_degree must be >= min_degree"
        assert all(op in ("+", "-") for op in self.operators), "invalid operator specified"


class SimpleIntegrationDataset(ProceduralDataset):
    """Generates simple integration problems with one variable"""

    def __init__(self, config: SimpleIntegrationConfig):
        self._prompt_templates = [
            "Find the indefinite integral: ∫ {integrand} dx",
            "Calculate the antiderivative: ∫ {integrand} dx",
            "Evaluate the indefinite integral: ∫ {integrand} dx",
        ]
        self.added_instruction = """
In addition, When doing calculation, Use the following instructions together with your mathematical ingenuity to solve the integral problems
## 1. Use ** instead ^ to represent powers. For example 7*X**2 instead of 7*X^2.
## 2. Always use * when doing all sorts of multiplcation in your reasoning steps. For example Use [-3*X**3*sin(X) - 9*X**2*cos(X) + 18*X*sin(X) + 18*cos(X) + C] instead of [-3x3sin(x) - 9x2cos(x) + 18xsin(x) + 18cos(x) + C].
"""
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _generate_coefficient(self, rng: random.Random) -> Fraction:
        """Generate a random coefficient for the polynomial"""
        if rng.choice([True, False]):  # 50% chance for integer
            return Fraction(rng.randint(self.config.min_bounds, self.config.max_bounds), 1)
        denominator = rng.randint(2, 10)
        return Fraction(rng.randint(self.config.min_bounds, self.config.max_bounds), denominator)

    def _generate_polynomial(self, rng: random.Random) -> tuple[sympy.Symbol, sympy.Expr]:
        """Generate a random polynomial with one variable"""
        terms = []
        x = sympy.Symbol(rng.choice(self.config.symbols))

        for _ in range(rng.randint(self.config.min_terms, self.config.max_terms)):
            coefficient = self._generate_coefficient(rng)
            degree = rng.randint(self.config.min_degree, self.config.max_degree)
            operator = rng.choice(self.config.operators)
            term = coefficient * x**degree
            if operator == "-":
                term = -term
            terms.append(term)
        return x, sum(terms)

    def __getitem__(self, idx: int) -> dict:
        rng = random.Random(self.seed + idx)
        symbol, polynomial = self._generate_polynomial(rng)
        derivative = sympy.diff(polynomial, symbol)
        question = rng.choice(self._prompt_templates).format(integrand=derivative) + self.added_instruction

        return {
            "question": question,
            "answer": str(polynomial) + " + C",
            "metadata": {
                "integrand": str(derivative),
                "variable": str(symbol),
                "expected_answer_expression": polynomial,
            },
        }

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """Determine if the solution provided solves the problem"""
        reward = 0.0
        metadata = entry["metadata"]
        if answer is not None:
            try:
                var = metadata["variable"]
                x = sympy.Symbol(var)
                # Parse answer while allowing integration constant 'C'
                user_expr = sympy.parse_expr(answer, local_dict={var: x, "C": sympy.Symbol("C")})
                # Compute derivative of student's answer
                derivative = sympy.diff(user_expr, x)
                integrand = sympy.parse_expr(metadata["integrand"], local_dict={var: x})

                # Check mathematical equivalence through simplification
                if sympy.simplify(derivative - integrand) == 0:
                    reward = 1.0
                elif answer.strip():
                    reward = 0.05
                else:
                    reward = 0.01
            except:
                reward = 0.01
        return reward


register_dataset("simple_integration", SimpleIntegrationDataset, SimpleIntegrationConfig)
