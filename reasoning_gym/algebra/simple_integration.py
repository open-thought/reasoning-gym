import random
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Optional

import sympy

from ..coaching import BaseCurriculum, RangeAttributeDefinition
from ..factory import ProceduralDataset, register_dataset

DATASET_NAME = "simple_integration"


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
When performing calculations, please follow these guidelines:
1. Use ** instead of ^ to represent exponents. For example, write 7*X**2 instead of 7*X^2.
2. Always include the * symbol for all multiplication operations in your reasoning steps. For example, write `-3*X**3*sin(X) - 9*X**2*cos(X) + 18*X*sin(X) + 18*cos(X) + C` instead of `-3x3sin(x) - 9x2cos(x) + 18xsin(x) + 18cos(x) + C`.
"""
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _generate_coefficient(self, rng: random.Random) -> Fraction:
        """Generate a random coefficient for the polynomial"""
        if rng.choice([True, False]):  # 50% chance for integer
            return Fraction(rng.randint(self.config.min_bounds, self.config.max_bounds), 1)
        denominator = rng.randint(2, 10)
        return Fraction(rng.randint(self.config.min_bounds, self.config.max_bounds), denominator)

    def _generate_polynomial(self, rng: random.Random, num_terms: int) -> tuple[sympy.Symbol, sympy.Expr]:
        """Generate a random polynomial with one variable"""
        terms = []
        x = sympy.Symbol(rng.choice(self.config.symbols))

        for _ in range(num_terms):
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
        num_terms = rng.randint(self.config.min_terms, self.config.max_terms)
        symbol, polynomial = self._generate_polynomial(rng, num_terms)
        derivative = sympy.diff(polynomial, symbol)
        question = rng.choice(self._prompt_templates).format(integrand=derivative) + self.added_instruction

        return {
            "question": question,
            "answer": str(polynomial) + " + C",
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "integrand": str(derivative),
                "variable": str(symbol),
                "num_terms": num_terms,
                "difficulty": {
                    "terms": (self.config.min_terms, self.config.max_terms),
                },
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
            except:
                reward = 0.0
        return reward


class SimpleIntegrationCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(SimpleIntegrationCurriculum.__name__, SimpleIntegrationConfig)
        self._define_attributes(
            RangeAttributeDefinition(
                name="terms",
                levels=[2, 3, 4, 5],
                lower_field_name="min_terms",
                upper_field_name="max_terms",
                description="The number of terms in the polynomial",
            )
        )


register_dataset(DATASET_NAME, SimpleIntegrationDataset, SimpleIntegrationConfig, SimpleIntegrationCurriculum)
