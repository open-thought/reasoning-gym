import cmath
import math
import random
from dataclasses import dataclass, field
from typing import Optional

from ..coaching import AttributeType, BaseCurriculum, ScalarAttributeDefinition
from ..factory import ProceduralDataset, register_dataset


@dataclass
class ComplexArithmeticConfig:
    min_real: int = -10
    max_real: int = 10
    min_imag: int = -10
    max_imag: int = 10
    operations: tuple[str, ...] = ("+", "-", "*", "/")
    operations_weights: list[float] = field(default_factory=lambda: [0.4, 0.4, 0.1, 0.1])
    seed: Optional[int] = None
    size: int = 500

    def validate(self) -> None:
        """Validate configuration parameters."""
        assert self.max_real >= self.min_real, "max_real must be >= min_real"
        assert self.max_imag >= self.min_imag, "max_imag must be >= min_imag"
        assert all(op in ("+", "-", "*", "/") for op in self.operations), "invalid operator"
        assert round(sum(self.operations_weights), 1) == 1.0, "operations_weights must sum to 1.0"


class ComplexArithmeticDataset(ProceduralDataset):
    """Generates complex number arithmetic problems."""

    def __init__(self, config: ComplexArithmeticConfig):
        self._prompt_templates = {
            "+": "Add the complex numbers: ({a}) + ({b})",
            "-": "Subtract the complex numbers: ({a}) - ({b})",
            "*": "Multiply the complex numbers: ({a}) × ({b})",
            "/": "Divide the complex numbers: ({a}) ÷ ({b})",
        }
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _generate_complex(self, rng: random.Random) -> complex:
        """Generate a random complex number."""
        real = rng.randint(self.config.min_real, self.config.max_real)
        imag = rng.randint(self.config.min_imag, self.config.max_imag)
        return complex(real, imag)

    def _format_complex(self, z: complex) -> str:
        """Format complex number with 2 decimal places."""
        real, imag = z.real, z.imag
        if abs(imag) < 1e-10:
            return f"{real:.2f}"
        elif abs(real) < 1e-10:
            return f"{imag:.2f}i"
        else:
            sign = "+" if imag >= 0 else "-"
            return f"{real} {sign} {abs(imag)}i"

    def __getitem__(self, idx: int) -> dict:
        rng = random.Random(self.seed + idx)

        # Choose random operation
        op = rng.choices(self.config.operations, weights=self.config.operations_weights, k=1)[0]

        if op == "/":
            # For division, first generate the quotient (a) and divisor (b)
            # Then calculate the dividend (result) as a * b
            a = self._generate_complex(rng)  # This will be the final result
            b = self._generate_complex(rng)
            while b == 0:  # Ensure non-zero divisor
                b = self._generate_complex(rng)
            result = a  # Store the intended result
            a = result * b  # Calculate dividend to ensure whole number division
        else:
            # For other operations, generate numbers normally
            a = self._generate_complex(rng)
            b = self._generate_complex(rng)

            # Calculate result
            if op == "+":
                result = a + b
            elif op == "-":
                result = a - b
            else:  # op == "*"
                result = a * b

        question = self._prompt_templates[op].format(a=self._format_complex(a), b=self._format_complex(b))

        return {
            "question": question,
            "answer": self._format_complex(result),
            "metadata": {
                "num1": (a.real, a.imag),
                "num2": (b.real, b.imag),
                "operation": op,
                "result": (int(result.real), int(result.imag)),  # Convert to int since we ensure whole numbers
                "difficulty": {
                    "min_real": self.config.min_real,
                    "max_real": self.config.max_real,
                    "min_imag": self.config.min_imag,
                    "max_imag": self.config.max_imag,
                    "operations_weights": self.config.operations_weights,
                },
            },
        }

    @staticmethod
    def parse_string_to_complex(answer: str) -> complex:
        if answer is None:
            return None

        try:
            # Normalize the answer string by removing spaces and converting to lowercase
            answer = answer.replace(" ", "").lower()

            # remove brackets
            while len(answer) > 1 and answer[0] == "(" and answer[-1] == ")":
                answer = answer[1:-1]

            # Convert mathematical notation 'i' to Python's 'j' for complex numbers
            answer = answer.replace("i", "j")

            # Handle real numbers (no imaginary part)
            if "j" not in answer:
                return complex(float(answer))

            # Handle pure imaginary numbers
            if answer == "j":
                return complex(0, 1)
            if answer == "-j":
                return complex(0, -1)

            # Handle cases like "7j" or "-7j" (no real part)
            if (
                answer.endswith("j")
                and not any(c in answer[:-1] for c in "+-")
                or (answer.startswith("-") and not any(c in answer[1:-1] for c in "+-"))
            ):
                # Extract coefficient
                coef = answer[:-1]
                if coef == "":
                    coef = "1"
                elif coef == "-":
                    coef = "-1"
                return complex(0, float(coef))

            # Handle complex numbers with both parts
            # Make sure there's a + or - before j if it's not at the beginning
            if "j" in answer and not answer.endswith("j"):
                return None  # Invalid format like "3j+2"

            # Handle cases like "3+j" (implicit 1)
            if "+j" in answer:
                answer = answer.replace("+j", "+1j")
            if "-j" in answer:
                answer = answer.replace("-j", "-1j")

            # Parse the normalized string into a complex number
            return complex(answer)

        except (ValueError, TypeError):
            return None

    def score_answer(self, answer: Optional[str], entry: dict) -> float:
        """Score the answer using exponential distance-based scoring."""
        if answer is None:
            return 0.0

        metadata = entry["metadata"]
        try:
            student_result = self.parse_string_to_complex(answer)
            expected_result = complex(*metadata["result"])
            # Calculate distance-based score using exponential decay
            distance = abs(student_result - expected_result)
            score = min(1.0, math.exp(-distance))  # Add 'import math' at the top
            return score

        except (ValueError, TypeError):
            return 0.0


class ComplexArithmeticCurriculum(BaseCurriculum):
    """Curriculum for complex number arithmetic problems."""

    def __init__(self):
        super().__init__(ComplexArithmeticCurriculum.__name__, ComplexArithmeticConfig)

        # Define attributes
        self._define_attributes(
            ScalarAttributeDefinition(
                name="min_real",
                field_name="min_real",
                levels=[-10, -100, -10000, -100000000],
                default_level=0,
                description="Minimum real part for complex numbers",
                attr_type=AttributeType.STATIC,
                min_value=-10,
            ),
            ScalarAttributeDefinition(
                name="max_real",
                field_name="max_real",
                levels=[10, 100, 10000, 100000000],
                default_level=0,
                description="Maximum real part for complex numbers",
                attr_type=AttributeType.STATIC,
                min_value=10,
            ),
            ScalarAttributeDefinition(
                name="min_imag",
                field_name="min_imag",
                levels=[-10, -100, -10000, -100000000],
                default_level=0,
                description="Minimum imaginary part for complex numbers",
                attr_type=AttributeType.STATIC,
                min_value=-10,
            ),
            ScalarAttributeDefinition(
                name="max_imag",
                field_name="max_imag",
                levels=[10, 100, 10000, 100000000],
                default_level=0,
                description="Maximum imaginary part for complex numbers",
                attr_type=AttributeType.STATIC,
                min_value=10,
            ),
            ScalarAttributeDefinition(
                name="operations_weights",
                field_name="operations_weights",
                levels=[[0.4, 0.4, 0.1, 0.1], [0.25, 0.25, 0.25, 0.25], [0.2, 0.2, 0.3, 0.3], [0.1, 0.1, 0.4, 0.4]],
                default_level=0,
                description="Operations weights to sample operation to use for each complex arithmetic problem",
                attr_type=AttributeType.STATIC,
                min_value=[0.4, 0.4, 0.1, 0.1],
            ),
        )


register_dataset("complex_arithmetic", ComplexArithmeticDataset, ComplexArithmeticConfig, ComplexArithmeticCurriculum)
