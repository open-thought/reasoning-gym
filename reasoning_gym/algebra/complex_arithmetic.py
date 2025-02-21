import cmath
import math
import random
from dataclasses import dataclass
from typing import Optional

from ..factory import ProceduralDataset, register_dataset


@dataclass
class ComplexArithmeticConfig:
    min_real: int = -10
    max_real: int = 10
    min_imag: int = -10
    max_imag: int = 10
    operations: tuple[str, ...] = ("+", "-", "*", "/")
    seed: Optional[int] = None
    size: int = 500

    def validate(self) -> None:
        """Validate configuration parameters."""
        assert self.max_real >= self.min_real, "max_real must be >= min_real"
        assert self.max_imag >= self.min_imag, "max_imag must be >= min_imag"
        assert all(op in ("+", "-", "*", "/") for op in self.operations), "invalid operator"


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
        op = rng.choice(self.config.operations)

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
            },
        }

    @staticmethod
    def parse_string_to_complex(answer: str) -> complex:
        try:
            # Normalize the answer string by removing spaces and converting to lowercase
            answer = answer.replace(" ", "").lower()
            # Convert mathematical notation 'i' to Python's 'j' for complex numbers
            answer = answer.replace("i", "j")

            # Handle real numbers (no imaginary part)
            if "j" not in answer:
                student_result = complex(float(answer))
            else:
                # Handle cases like "j" or "2j" (implicit coefficient)
                if answer[0] == "j":
                    # Convert "j" to "1j", "2j" remains unchanged
                    answer = "1" + answer
                # Handle cases like "3j" where there's no explicit + or - before j
                elif answer[-1] == "j" and not any(c in answer[:-1] for c in "+-"):
                    # Convert "3j" to "3+1j"
                    answer = answer.replace("j", "+1j")

                # Ensure the string has an imaginary part, even if zero
                if "j" not in answer:
                    answer += "+0j"

                # Parse the normalized string into a complex number
                student_result = complex(answer)

        except ValueError:
            return None

        return student_result

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


register_dataset("complex_arithmetic", ComplexArithmeticDataset, ComplexArithmeticConfig)
