import random
import string
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import sympy as sp

from ..factory import ProceduralDataset, register_dataset


@dataclass
class PolynomialMultiplicationConfig:
    """
    Configuration for polynomial multiplication task generation.
    """

    min_terms: int = 2  # Minimum number of polynomial terms
    max_terms: int = 4  # Maximum number of polynomial terms
    min_value: int = 1  # Minimum value for coefficients
    max_value: int = 100  # Maximum value for coefficients
    min_degree: int = 1  # Minimum polynomial degree
    max_degree: int = 3  # Maximum polynomial degree
    min_polynomials: int = 2  # Minimum number of polynomials being multiplied
    max_polynomials: int = 3  # Maximum number of polynomials being multiplied
    single_variable: bool = True
    operators: Tuple[str, ...] = (
        "+",
        "-",
    )  # Allowed operators between terms, Avoid adding '*' or '/' because they will affect the degree
    seed: Optional[int] = None
    size: int = 500

    def validate(self) -> None:
        """Validate configuration parameters."""
        assert self.min_terms > 0, "min_terms must be positive."
        assert self.max_terms >= self.min_terms, "max_terms must be >= min_terms."

        assert self.min_value > 0, "min_value must be positive."
        assert self.max_value >= self.min_value, "max_value must be >= min_value."

        assert self.min_degree >= 1, "min_degree must be >= 1."
        assert self.max_degree >= self.min_degree, "max_degree must be >= min_degree."

        assert self.min_polynomials >= 2, "min_polynomials must be >= 2."
        assert self.max_polynomials >= self.min_polynomials, "max_polynomials must be >= min_polynomials."

        allowed_ops = {"+", "-"}
        assert len(self.operators) > 0, "operators tuple cannot be empty."
        assert all(op in allowed_ops for op in self.operators), "Invalid operator found. Must be a subset of {+, -}."


class PolynomialMultiplicationDataset(ProceduralDataset):
    """
    Generates [min_polynomials, max_polynomials] random polynomials of degree in [min_degree, max_degree].
    - The polynomial is formed by summing random terms of the form: coeff * x^exponent.
    - Then we find "F = P_0 * ... * P_1" using Sympy.
    """

    def __init__(self, config: PolynomialMultiplicationConfig):
        self._prompt_templates = [
            "Simplify this expression: {polynomial_expr}",
            "Calculate the following: {polynomial_expr}",
        ]
        super().__init__(config=config, seed=config.seed, size=config.size)

    def __getitem__(self, idx: int) -> dict:
        """
        Generate a single polynomial multiplication item.

        Returns:
            A dict with:
                - question: str (e.g. "Multiply polynomials: (8x^3 + x + 2)*(x - 3)")
                - answer: str (Product, e.g. "8x^4 - 24x^3 + x^2 - x - 6")
                - metadata: dict with details (polynomial_expr, single_variable)
        """
        rng = random.Random(self.seed + idx)
        number_polynomials = rng.randint(self.config.min_polynomials, self.config.max_polynomials)
        polynomials = [self._generate_polynomial_expr(rng) for i in range(number_polynomials)]

        polynomial_expr = sp.prod(polynomials)
        product = sp.expand(polynomial_expr)

        return {
            "question": rng.choice(self._prompt_templates).format(
                polynomial_expr=polynomial_expr,
            ),
            "answer": product,
            "metadata": {
                "polynomial_expr": str(polynomial_expr),
                "single_variable": self.config.single_variable,
                "result": str(product),
            },
        }

    def _get_variable(self, rng: random.Random) -> str:
        """Get a random lowercase variable name"""
        if self.config.single_variable:
            return "x"
        return rng.choice(string.ascii_lowercase)

    def _generate_polynomial_expr(self, rng: random.Random):
        """
        Randomly generate a polynomial expression of 'degree'.
        We'll use the config parameters:
            - min_terms, max_terms: how many total terms to combine
            - min_value, max_value: range for coefficients
            - operators: to decide sign flips or direct addition

         Args:
            rng: Random number generator

        Returns:
            Polynomial string
        """
        variable = self._get_variable(rng)
        degree = rng.randint(self.config.min_degree, self.config.max_degree)

        x = sp.Symbol(variable)

        # Choose the number of terms and their respective degrees
        num_terms = rng.randint(self.config.min_terms, self.config.max_terms)
        # Keep track of exponents, exponents can repeat or skip but we force the highest exponent
        chosen_exponents = [degree]
        # Fill the rest randomly in [0, degree]
        for _ in range(num_terms - 1):
            exp = rng.randint(0, degree)
            chosen_exponents.append(exp)

        # Now build the polynomial expression: sum_{term}( coeff * x^exponent ), with optional sign
        polynomial_expr = 0
        for exp in chosen_exponents:
            coeff = rng.randint(self.config.min_value, self.config.max_value)
            # If '-' in operators, we can randomly flip the sign
            if "-" in self.config.operators and rng.random() < 0.5:
                coeff = -coeff
            term_expr = coeff * (x**exp)
            polynomial_expr += term_expr

        return polynomial_expr

    def score_answer(self, answer: Optional[str], entry: Dict[str, Any]) -> float:
        reward = 0.0
        metadata = entry["metadata"]
        if answer is not None:
            try:
                predicted_poly = sp.parse_expr(answer)
                target_poly = sp.parse_expr(metadata["result"])

                # Check if the difference simplifies to zero (i.e. they are equivalent).
                if sp.simplify(predicted_poly - target_poly) == 0:
                    reward = 1.0
                elif answer.strip():
                    reward = 0.05
                else:
                    reward = 0.01
            except Exception:
                reward = 0.01
        return reward


register_dataset("polynomial_multiplication", PolynomialMultiplicationDataset, PolynomialMultiplicationConfig)
