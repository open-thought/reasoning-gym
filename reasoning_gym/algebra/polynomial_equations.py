import random
import string
from dataclasses import dataclass
from typing import Optional, Tuple

from sympy import Eq, Symbol, expand, solve

from ..factory import ProceduralDataset, register_dataset


@dataclass
class PolynomialEquationsConfig:
    """
    Configuration for polynomial equation task generation.
    """

    min_terms: int = 2  # Minimum number of polynomial terms
    max_terms: int = 4  # Maximum number of polynomial terms
    min_value: int = 1  # Minimum value for coefficients
    max_value: int = 100  # Maximum value for coefficients
    min_degree: int = 1  # Minimum polynomial degree
    max_degree: int = 3  # Maximum polynomial degree
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

        allowed_ops = {"+", "-"}
        assert len(self.operators) > 0, "operators tuple cannot be empty."
        assert all(op in allowed_ops for op in self.operators), "Invalid operator found. Must be a subset of {+, -}."


class PolynomialEquationsDataset(ProceduralDataset):
    """
    Generates random polynomial equations of degree in [min_degree, max_degree].
    - The polynomial is formed by summing random terms of the form: coeff * x^exponent.
    - Then we solve "polynomial_expr = 0" using Sympy.
    - The solution may be real or complex; we filter real solutions by default for simplicity.
    """

    def __init__(self, config: PolynomialEquationsConfig):
        self._prompt_templates = [
            "Find the real value(s) of {variable} in the equation: {polynomial_expanded} = 0",
            "Solve for real {variable}: {polynomial_expanded} = 0",
            "Determine the real value(s) of {variable} tha satisfies: {polynomial_expanded} = 0",
            "Solve the polynomial equation for real {variable}:\n{polynomial_expanded} = 0",
        ]
        super().__init__(config=config, seed=config.seed, size=config.size)

    def __getitem__(self, idx: int) -> dict:
        """
        Generate a single polynomial equation item.

        Returns:
            A dict with:
                - question: str (e.g. "Solve the polynomial equation: 2*x^2 - 3*x + 1 = 0")
                - answer: str (the sorted list of real solutions, e.g. "[0.5, 1.0]")
                - metadata: dict with details (polynomial_expr, degree, etc.)
        """
        rng = random.Random(self.seed + idx)

        # Get variable and generate polynomial equation in standard form
        variable = self._get_variable(rng)
        degree = rng.randint(self.config.min_degree, self.config.max_degree)
        polynomial_expr = self._generate_polynomial_expr(rng, variable, degree)
        polynomial_expanded = expand(polynomial_expr)

        # Solve the polynomial = 0
        # We filter real solutions only
        solutions = solve(Eq(polynomial_expanded, 0), variable, dict=False)
        real_solutions = []
        for sol in solutions:
            if sol.is_real:
                # Evaluate symbolic solution to a floating approximation
                real_solutions.append(float(sol.evalf()))
        real_solutions.sort()
        answer_str = str(real_solutions)

        return {
            "question": rng.choice(self._prompt_templates).format(
                variable=variable,
                polynomial_expanded=polynomial_expanded,
            ),
            "answer": answer_str,
            "metadata": {
                "polynomial_expr": str(polynomial_expanded),
                "variable": variable,
                "degree": degree,
                "real_solutions": real_solutions,
            },
        }

    def _get_variable(self, rng: random.Random) -> str:
        """Get a random lowercase variable name"""
        return rng.choice(string.ascii_lowercase)

    def _generate_polynomial_expr(self, rng: random.Random, variable: Symbol, degree: int):
        """
        Randomly generate a polynomial expression of 'degree'.
        We'll use the config parameters:
            - min_terms, max_terms: how many total terms to combine
            - min_value, max_value: range for coefficients
            - operators: to decide sign flips or direct addition

         Args:
            rng: Random number generator
            variable: Variable symbol to use in equation
            degree: Highest degree. We ensure that there is at least one term with exponent=degree

        Returns:
            Polynomial string
        """
        x = Symbol(variable)

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


register_dataset("polynomial_equations", PolynomialEquationsDataset, PolynomialEquationsConfig)
