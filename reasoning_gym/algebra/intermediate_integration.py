import random
from dataclasses import dataclass
from typing import Any, Optional

import sympy

from ..factory import ProceduralDataset, register_dataset


@dataclass
class IntermediateIntegrationConfig:
    problem_types: tuple = ("substitution", "by_parts")
    substitution_types: tuple = (
        "linear",  # (ax + b)^n
        "trigonometric",  # sin**2(x)cos(x)
        "exponential",  # 2xe^x**2
        "radical",  # x (3x + 2)^1/2
    )

    # Integration by parts problem categories
    by_parts_types: tuple = (
        "polynomial_exp_trig",  # e.g. x^2*e^x
        "log_inverse_trig",  # e.g. ln(x)/arctan(x)
        "cyclic",  # e.g. e^x*sinx requiring cyclic integration
        "repeated_parts",  # Requires multiple integration by parts
    )
    seed: Optional[int] = None
    size: int = 500

    linear_lower_bound: int = 1  # coefficient of linear expression
    linear_upper_bound: int = 10
    min_linear_degree: int = 2  # degree of linear expression in substitution problem
    max_linear_degree: int = 4
    outer_constant_min: int = 1  # multiplicative constant to multiply integrand by
    outer_constant_max: int = 3
    min_poly_degree: int = 1  # degree of polynomial in by parts problem
    max_poly_degree: int = 3
    symbols: tuple = ("x", "X")
    operators: tuple = (
        "+",
        "-",
    )

    def validate(self) -> None:
        """Validate the configuration parameters of the integral problem"""
        assert self.size > 0, "size must be positive"
        assert self.linear_lower_bound > 0, "linear_lower_bound must be positive"
        assert self.linear_upper_bound >= self.linear_lower_bound, "linear_upper_bound must be >= linear_lower_bound"
        assert self.min_linear_degree > 0, "min_linear_degree must be positive"
        assert self.max_linear_degree >= self.min_linear_degree, "max_linear_degree must be >= min_linear_degree"
        assert self.outer_constant_min > 0, "outer_constant_min must be positive"
        assert self.outer_constant_max >= self.outer_constant_min, "outer_constant_max must be >= outer_constant_min"
        assert self.min_poly_degree > 0, "min_poly_degree must be positive"
        assert self.max_poly_degree >= self.min_poly_degree, "max_poly_degree must be >= min_poly_degree"
        assert all(op in ("+", "-") for op in self.operators), "invalid operator specified"
        assert all(symbols in ("x", "X") for symbols in self.symbols), "invalid symbol specified"
        assert all(t in ("substitution", "by_parts") for t in self.problem_types), "invalid problem type"
        assert all(
            t in ("linear", "trigonometric", "exponential", "radical") for t in self.substitution_types
        ), "invalid substitution type"
        assert all(
            t in ("polynomial_exp_trig", "log_inverse_trig", "cyclic", "repeated_parts") for t in self.by_parts_types
        ), "invalid by_parts type"


class IntermediateIntegrationDataset(ProceduralDataset):
    """Generates intermediate integration problem - either
    by substitution or by parts"""

    """Add multiplicative constant"""

    def __init__(self, config: IntermediateIntegrationConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)
        self.prompt_template = [
            "Find the indefinite integral: ∫ {integrand} dx",
            "Calculate the antiderivative: ∫ {integrand} dx",
            "Evaluate the indefinite integral: ∫ {integrand} dx",
        ]
        self.added_instruction = """
In addition, when doing calculation, use the following instructions together with your mathematical ingenuity to solve the integral problems
## 1. Use ** instead ^ to represent powers. For example 7*X**2 instead of 7*X^2.
## 2. Always use * when doing all sorts of multiplcation in your reasoning steps. For example Use [-3*X**3*sin(X) - 9*X**2*cos(X) + 18*X*sin(X) + 18*cos(X) + C] instead of [-3x3sin(x) - 9x2cos(x) + 18xsin(x) + 18cos(x) + C].
"""

    def _get_outer_constant(self, rng: random.Random) -> int:
        """Helper to generate signed outer constant from config"""
        value = rng.randint(self.config.outer_constant_min, self.config.outer_constant_max)
        return -value if rng.choice(self.config.operators) == "-" else value

    def _generate_linear_substitution_problem(self, rng: random.Random, x: sympy.Symbol) -> sympy.Expr:
        """Generate a linear substitution problem with outer constant"""
        a = rng.randint(self.config.linear_lower_bound, self.config.linear_upper_bound)
        b = rng.randint(self.config.linear_lower_bound, self.config.linear_upper_bound)

        linear_function = a * x + (b if rng.choice(self.config.operators) == "+" else -b)
        degree = rng.randint(self.config.min_linear_degree, self.config.max_linear_degree)

        return self._get_outer_constant(rng) * linear_function**degree

    def _generate_exponential_substitution(self, rng: random.Random, x: sympy.Symbol) -> sympy.Expr:
        """Generate exponential substitution problem with outer constant"""
        exponent_type = rng.choice(["linear", "quadratic"])

        # Generate terms with signs
        num_terms = 2 if exponent_type == "linear" else 3
        terms = [
            (-1 if rng.choice(self.config.operators) == "-" else 1)
            * rng.randint(self.config.linear_lower_bound, self.config.linear_upper_bound)
            for _ in range(num_terms)
        ]

        if exponent_type == "linear":
            u = terms[0] * x + terms[1]
            du_dx = terms[0]
        else:  # Quadratic
            u = terms[0] * x**2 + terms[1] * x + terms[2]
            du_dx = 2 * terms[0] * x + terms[1]

        return self._get_outer_constant(rng) * du_dx * sympy.exp(u)

    def _generate_radical_substitution(self, rng: random.Random, x: sympy.Symbol) -> sympy.Expr:
        """Generate radical substitution problem with outer constant"""

        # Generate linear expression under radical: ax + b with possible negative coefficients
        a = (-1 if rng.choice(self.config.operators) == "-" else 1) * rng.randint(
            self.config.linear_lower_bound, self.config.linear_upper_bound
        )
        b = (-1 if rng.choice(self.config.operators) == "-" else 1) * rng.randint(
            self.config.linear_lower_bound, self.config.linear_upper_bound
        )

        u = a * x + b
        derivative = a  # du/dx

        integrand = derivative * sympy.sqrt(u)
        return self._get_outer_constant(rng) * integrand

    def _generate_trigonometric_substitution(self, rng: random.Random, x: sympy.Symbol) -> sympy.Expr:
        """Generate trigonometric substitution with outer constant"""
        trig_func = rng.choice(["sin", "cos"])

        # Generate signed coefficients
        a = (-1 if rng.choice(self.config.operators) == "-" else 1) * rng.randint(
            self.config.linear_lower_bound, self.config.linear_upper_bound
        )
        b = (-1 if rng.choice(self.config.operators) == "-" else 1) * rng.randint(
            self.config.linear_lower_bound, self.config.linear_upper_bound
        )

        inner = a * x + b
        power = rng.randint(1, 4)
        if trig_func == "sin":
            integrand = a * sympy.cos(inner) * sympy.sin(inner) ** power
        else:
            integrand = -a * sympy.sin(inner) * sympy.cos(inner) ** power
        return self._get_outer_constant(rng) * integrand

    def _generate_polynomial_exp_trig(self, rng: random.Random, x: sympy.Symbol) -> sympy.Expr:
        """Generate polynomial × exponential/trigonometric integrand"""
        poly_degree = rng.randint(self.config.min_poly_degree, self.config.max_poly_degree)

        func_type = rng.choice(["exp", "sin", "cos"])
        if func_type == "exp":
            transcendental = sympy.exp(x)
        else:
            coefficient = rng.randint(1, 3)
            transcendental = sympy.Function(func_type)(coefficient * x)

        polynomial = x**poly_degree
        integrand = polynomial * transcendental
        return self._get_outer_constant(rng) * integrand

    def _generate_log_inverse_trig(self, rng: random.Random, x: sympy.Symbol) -> sympy.Expr:
        """Generate logarithmic or inverse trigonometric integrand"""
        func_type = rng.choice(["log", "asin", "atan"])

        if func_type == "log":
            log_arg = x if rng.random() < 0.8 else x ** rng.randint(2, 3)
            func = sympy.ln(log_arg)
        else:
            coefficient = rng.randint(1, 3)
            func = sympy.Function(func_type)(coefficient * x)

        return self._get_outer_constant(rng) * func

    def _generate_cyclic_integral(self, rng: random.Random, x: sympy.Symbol) -> sympy.Expr:
        """Generate cyclic integral (e.g., e^x * sinx)"""
        func_pair = rng.choice(
            [(sympy.exp(x), sympy.sin(x)), (sympy.exp(x), sympy.cos(x)), (sympy.sin(x), sympy.cos(x))]
        )
        integrand = func_pair[0] * func_pair[1]
        return self._get_outer_constant(rng) * integrand

    def _generate_repeated_parts(self, rng: random.Random, x: sympy.Symbol):
        """Generate problem requiring multiple integration by parts"""
        poly_degree = rng.randint(3, self.config.max_poly_degree)
        transcendental = rng.choice([sympy.sin(x), sympy.cos(x), sympy.exp(x)])
        integrand = x**poly_degree * transcendental
        return self._get_outer_constant(rng) * integrand

    def __getitem__(self, index: int):
        """Generate either substitution or by-parts problem"""
        rng = random.Random(self.seed + index)
        problem_type = rng.choice(self.config.problem_types)
        x = sympy.Symbol(rng.choice(self.config.symbols))

        if problem_type == "substitution":
            substitution_type = rng.choice(self.config.substitution_types)
            if substitution_type == "linear":
                integrand = self._generate_linear_substitution_problem(rng, x)
            elif substitution_type == "trigonometric":
                integrand = self._generate_trigonometric_substitution(rng, x)
            elif substitution_type == "exponential":
                integrand = self._generate_exponential_substitution(rng, x)
            elif substitution_type == "radical":
                integrand = self._generate_radical_substitution(rng, x)
        else:
            parts_type = rng.choice(self.config.by_parts_types)
            if parts_type == "polynomial_exp_trig":
                integrand = self._generate_polynomial_exp_trig(rng, x)
            elif parts_type == "log_inverse_trig":
                integrand = self._generate_log_inverse_trig(rng, x)
            elif parts_type == "cyclic":
                integrand = self._generate_cyclic_integral(rng, x)
            elif parts_type == "repeated_parts":
                integrand = self._generate_repeated_parts(rng, x)

        answer = sympy.integrate(integrand, x)
        answer_str = str(answer) + " + C"
        question = rng.choice(self.prompt_template).format(integrand=integrand) + self.added_instruction

        return {
            "question": question,
            "answer": answer_str,
            "metadata": {
                "integrand": str(integrand),
                "problem_type": problem_type,
                "variable": str(x),
                "type": substitution_type if problem_type == "substitution" else parts_type,
                "expected_answer_expression": answer,
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


register_dataset("intermediate_integration", IntermediateIntegrationDataset, IntermediateIntegrationConfig)
