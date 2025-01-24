from dataclasses import dataclass
from random import Random
from typing import Any, Literal, Optional

from ..dataset import ProceduralDataset


@dataclass
class BasicArithmeticDatasetConfig:
    """Configuration for arithmetic dataset generation"""

    min_terms: int = 2
    max_terms: int = 6
    min_digits: int = 1
    max_digits: int = 4
    operators: list[str] = ("+", "-", "*", "/")
    allow_parentheses: bool = True
    allow_negation: bool = True
    seed: Optional[int] = None
    size: int = 500  # Virtual dataset size
    format_style: Literal["simple", "natural"] = "simple"
    whitespace: Literal["no_space", "single", "random"] = "single"  # Whitespace style between terms

    def validate(self):
        """Validate configuration parameters"""
        assert self.min_terms > 0, "min_terms must be positive"
        assert self.max_terms >= self.min_terms, "max_terms must be >= min_terms"
        assert self.min_digits > 0, "min_digits must be positive"
        assert self.max_digits >= self.min_digits, "max_digits must be >= min_digits"
        assert len(self.operators) > 0, "must provide at least one operator"
        for op in self.operators:
            assert op in ["+", "-", "*", "/"], f"unsupported operator: {op}"


def find_common_divisors(a: int, b: int) -> list[int]:
    # Helper function to find GCD using Euclidean algorithm
    def gcd(x, y):
        while y:
            x, y = y, x % y
        return x

    # Get the GCD of the two numbers
    gcd_value = gcd(abs(a), abs(b))
    # Find all divisors of the GCD
    divisors = []
    i = 1
    # We only need to check up to sqrt(gcd_value)
    while i * i <= gcd_value:
        if gcd_value % i == 0:
            divisors.append(i)
            # Don't add the same number twice for perfect squares
            if i * i != gcd_value:
                divisors.append(gcd_value // i)
        i += 1
    return divisors


def eval_floordiv(exp: str) -> int:
    return eval(exp.replace("/", "//"))


class BasicArithmeticDataset(ProceduralDataset):
    """Dataset that generates basic arithmetic tasks with configurable complexity"""

    def __init__(self, config: BasicArithmeticDatasetConfig):
        self.config = config
        self.config.validate()
        super().__init__(seed=config.seed, size=config.size)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Generate a single arithmetic task

        Args:
            idx: Index of the item to generate

        Returns:
            dict with keys:
                - question: str, the formatted arithmetic expression
                - answer: str, the ground truth result
                - metadata: dict with generation parameters
        """
        # Create deterministic RNG from base seed and idx
        item_rng = Random(self.seed + idx)

        num_terms = item_rng.randint(self.config.min_terms, self.config.max_terms)
        num_digits = item_rng.randint(self.config.min_digits, self.config.max_digits)

        if self.config.allow_parentheses:
            expression, result = self._generate_complex_task(item_rng, num_terms, num_digits)
        else:
            expression, result = self._generate_simple_task(item_rng, num_terms, num_digits)

        question = self._format_question(item_rng, expression)

        return {
            "question": question,
            "answer": str(result),
            "metadata": {
                "num_terms": num_terms,
                "num_digits": num_digits,
                "expression": expression,
            },
        }

    def _generate_complex_task(self, rng: Random, num_terms: int, num_digits: int) -> tuple[str, int]:
        """Generate a complex arithmetic task with possible parentheses"""

        def add_terms(remaining: int) -> list[str]:
            # split terms randomly into left and right
            num_left = rng.randint(1, remaining)
            num_right = remaining - num_left

            left_parts = []
            if num_left > 1 and rng.random() > 0.5 and self.config.allow_parentheses:
                if rng.random() > 0.5 and self.config.allow_negation:
                    left_parts.append("-(")
                else:
                    left_parts.append("(")
                left_parts.extend(add_terms(num_left))
                left_parts.append(")")
            else:
                for i in range(num_left):
                    c = rng.randint(-(10**num_digits) + 1, 10**num_digits - 1)
                    left_parts.append(str(c))
                    if i + 1 < num_left:
                        left_parts.append(rng.choice([o for o in self.config.operators if o != "/"]))

            if num_right == 0:
                return left_parts

            op = rng.choice(self.config.operators)
            if op != "/":
                left_parts.append(op)
                left_parts.extend(add_terms(num_right))
            else:
                # left part has parantheses or no division
                dividend = eval_floordiv("".join(left_parts) if left_parts[-1] == ")" else left_parts[-1])
                left_parts.append(op)

                if num_right > 1:
                    right_parts = add_terms(num_right - 1)
                    if right_parts[-1] == ")":
                        right_value = eval_floordiv("".join(right_parts))

                        if right_value == 0:
                            correction = 1
                        else:
                            target = rng.choice(find_common_divisors(dividend, right_value))
                            correction = target - right_value

                        right_parts.pop()
                        right_parts.append("+")
                        right_parts.append(str(correction))
                        right_parts.append(")")

                    else:
                        divisor = rng.choice(find_common_divisors(dividend, 0))
                        left_parts.append(str(divisor))
                        left_parts.append("+")

                    left_parts.extend(right_parts)
                else:
                    if dividend != 0:
                        divisor = rng.choice(find_common_divisors(dividend, 0))
                    else:
                        divisor = rng.randint(1, 10**num_digits - 1)
                    left_parts.append(str(divisor))

            return left_parts

        parts = add_terms(num_terms)

        # Add whitespace according to config
        if self.config.whitespace == "no_space":
            expression = "".join(parts)
        elif self.config.whitespace == "single":
            expression = " ".join(parts)
        else:  # random
            space_parts = []
            for p in parts:
                if rng.random() < 0.15:
                    space_parts.append(" ")
                space_parts.append(p)
            expression = "".join(space_parts).strip()
        result = eval_floordiv(expression)  # Note: eval is safe here as we control the input

        return expression, result

    def _generate_simple_task(self, rng: Random, num_terms: int, num_digits: int) -> tuple[str, int]:
        """Generate a simple linear arithmetic task without parentheses"""
        constants = [rng.randint(0, 10**num_digits) for _ in range(num_terms)]
        operators = [rng.choice(self.config.operators) for _ in range(num_terms - 1)]

        # Build expression and compute result
        expression_parts = []
        result = constants[0]

        expression_parts.append(str(constants[0]))
        for i, op in enumerate(operators):
            c = constants[i + 1]
            expression_parts.append(op)
            expression_parts.append(str(c))

            if op == "+":
                result += c
            elif op == "-":
                result -= c
            elif op == "*":
                result *= c
            elif op == "/":
                # Find a number that divides result evenly
                divisors = [d for d in range(2, min(abs(result), 10**num_digits)) if result % d == 0]
                if divisors:
                    c = rng.choice(divisors)
                    result //= c
                else:
                    # Fallback to multiplication if no clean division possible
                    op = "*"
                    c = rng.randint(1, 10**num_digits - 1)
                    result *= c
            else:
                raise RuntimeError(f"Unsupported operator: {op}")

        expression = " ".join(expression_parts)
        return expression, result

    def _format_question(self, rng: Random, expression: str) -> str:
        """Format the expression according to config style"""
        if self.config.format_style == "simple":
            return f"{expression} ="
        else:
            templates = ["What is {0}?", "Calculate {0}", "Solve {0}", "Evaluate the expression: {0}"]
            return rng.choice(templates).format(expression)


def basic_arithmetic_dataset(
    min_terms: int = 2,
    max_terms: int = 6,
    min_digits: int = 1,
    max_digits: int = 4,
    operators: list[str] = ("+", "-", "*", "/"),
    allow_parentheses: bool = True,
    allow_negation: bool = True,
    seed: Optional[int] = None,
    size: int = 500,
    format_style: Literal["simple", "natural"] = "simple",
    whitespace: Literal["no_space", "single", "random"] = "single",
) -> BasicArithmeticDataset:
    """Create a BasicArithmeticDataset with the given configuration.

    Args:
        min_terms: Minimum number of terms in expressions
        max_terms: Maximum number of terms in expressions
        min_digits: Minimum number of digits in numbers
        max_digits: Maximum number of digits in numbers
        operators: List of operators to use ("+", "-", "*")
        allow_parentheses: Whether to allow parentheses in expressions
        allow_negation: Whether to allow negative numbers
        seed: Random seed for reproducibility
        size: Virtual size of the dataset
        format_style: Style of question formatting ("simple" or "natural")

    Returns:
        BasicArithmeticDataset: Configured dataset instance
    """
    config = BasicArithmeticDatasetConfig(
        min_terms=min_terms,
        max_terms=max_terms,
        min_digits=min_digits,
        max_digits=max_digits,
        operators=operators,
        allow_parentheses=allow_parentheses,
        allow_negation=allow_negation,
        seed=seed,
        size=size,
        format_style=format_style,
        whitespace=whitespace,
    )
    return BasicArithmeticDataset(config)
