import random
import warnings
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
    variables: Tuple[str] = ("x", "y", "z")  # Tuple of variable names, that will be chosen randomly
    allow_cross_variable_product: bool = False  # Generate tasks like "Multiply (x^2+3x-1)*(y^2-5)"
    allow_multivariate_polynomials: bool = False  # Generate multivariate tasks like "Multiply (2x^2 + 3y)*(5x^2+3x-1)"
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

        assert len(self.variables) > 0, "The variable tuple is empty."
        assert not (
            self.allow_multivariate_polynomials and not self.allow_cross_variable_product
        ), "Multivariate polynomials require cross product."

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
                - metadata: dict with details (polynomial_expr, result, variables)
        """
        rng = random.Random(self.seed + idx)

        polynomial_expr = sp.prod(self._generate_polynomial_product(rng))
        product = sp.expand(polynomial_expr)

        return {
            "question": rng.choice(self._prompt_templates).format(
                polynomial_expr=polynomial_expr,
            ),
            "answer": product,
            "metadata": {
                "polynomial_expr": str(polynomial_expr),
                "result": str(product),
                "variables": list(product.free_symbols),
            },
        }

    def _get_variable(self, rng: random.Random) -> str:
        """Get a random lowercase variable name"""
        return rng.choice(self.config.variables)

    def _generate_polynomial_product(self, rng):
        """Helper for selecting regular or multivariate polynomial. Returns expressions and unique variables."""

        variable = None if self.config.allow_cross_variable_product else self._get_variable(rng)
        number_polynomials = rng.randint(self.config.min_polynomials, self.config.max_polynomials)

        if self.config.allow_multivariate_polynomials:
            generated = [self._generate_multivariate_polynomial(rng) for _ in range(number_polynomials)]
        else:
            generated = [self._generate_regular_polynomial(rng, variable) for _ in range(number_polynomials)]

        return generated

    def _generate_multivariate_polynomial(self, rng: random.Random):
        """Generates a multivariate polynomial, returns variable set and expression."""
        # Choose the number of terms and their respective degrees
        num_terms = rng.randint(self.config.min_terms, self.config.max_terms)

        polynomial_expr = 0
        for _ in range(num_terms):
            # Pick a nonzero random coefficient between min_value and max_value.
            coeff = rng.randint(self.config.min_value, self.config.max_value)

            # Build the monomial by choosing each exponent independently.
            monomial = 1
            var = self._get_variable(rng)
            for v in var:
                v = sp.Symbol(v)
                exp = random.randint(self.config.min_degree, self.config.max_degree)
                monomial *= v**exp

            # If '-' in operators, we can randomly flip the sign
            if "-" in self.config.operators and rng.random() < 0.5:
                coeff = -coeff

            polynomial_expr += coeff * monomial

        return polynomial_expr

    def _generate_regular_polynomial(self, rng: random.Random, variable: Optional[str]):
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
        variable = variable if variable else self._get_variable(rng)
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
                predicted_poly = sp.Poly(answer)
                target_poly = sp.Poly(metadata["result"])

                # Check if the difference simplifies to zero (i.e. they are equivalent).
                if predicted_poly == target_poly:
                    reward = 1.0
                elif answer.strip():
                    reward = 0.05
                else:
                    reward = 0.01
            except Exception:
                reward = 0.01
        return reward


register_dataset("polynomial_multiplication", PolynomialMultiplicationDataset, PolynomialMultiplicationConfig)
