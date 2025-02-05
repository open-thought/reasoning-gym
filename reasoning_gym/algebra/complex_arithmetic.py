import random
from dataclasses import dataclass
from typing import Optional, Tuple
import cmath

from ..factory import ProceduralDataset, register_dataset


@dataclass
class ComplexArithmeticConfig:
    min_real: int = -10
    max_real: int = 10
    min_imag: int = -10
    max_imag: int = 10
    operations: Tuple[str, ...] = ("+", "-", "*", "/")
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
        """Format complex number for display."""
        real, imag = z.real, z.imag
        if imag == 0:
            return f"{real:.0f}"
        elif real == 0:
            return f"{imag:.0f}i"
        else:
            sign = "+" if imag >= 0 else "-"
            return f"{real:.0f} {sign} {abs(imag):.0f}i"

    def __getitem__(self, idx: int) -> dict:
        rng = random.Random(self.seed + idx)
        
        # Generate two random complex numbers
        a = self._generate_complex(rng)
        b = self._generate_complex(rng)
        
        # For division, ensure denominator is not zero
        while b == 0:
            b = self._generate_complex(rng)

        # Choose random operation
        op = rng.choice(self.config.operations)
        
        # Calculate result
        if op == "+":
            result = a + b
        elif op == "-":
            result = a - b
        elif op == "*":
            result = a * b
        else:  # op == "/"
            result = a / b

        question = self._prompt_templates[op].format(
            a=self._format_complex(a),
            b=self._format_complex(b)
        )

        return {
            "question": question,
            "answer": self._format_complex(result),
            "metadata": {
                "num1": (a.real, a.imag),
                "num2": (b.real, b.imag),
                "operation": op,
                "result": (result.real, result.imag),
            },
        }

    def score_answer(self, answer: str, metadata: dict) -> float:
        """Score the answer, allowing for minor formatting differences."""
        if answer is None:
            return 0.0

        try:
            # Convert the expected result from metadata
            expected_result = complex(*metadata["result"])
            
            # Parse student answer
            # Remove spaces and convert to lowercase
            answer = answer.replace(" ", "").lower()
            
            # Handle different forms of writing complex numbers
            if "i" not in answer and "j" not in answer:
                # Pure real number
                return abs(complex(float(answer)) - expected_result) < 1e-10

            # Replace 'i' with 'j' for Python's complex number notation
            answer = answer.replace('i', 'j')
            
            # Handle cases like "2j" (add plus sign)
            if answer[0] == 'j':
                answer = '1' + answer
            elif answer[-1] == 'j' and not any(c in answer[:-1] for c in '+-'):
                answer = answer.replace('j', '+1j')
            
            # Add missing real or imaginary parts
            if 'j' not in answer:
                answer += '+0j'
            
            # Parse the answer string into a complex number
            student_result = complex(answer)
            
            # Check if the results are close enough (allowing for minor floating-point differences)
            return float(abs(student_result - expected_result) < 1e-10)

        except (ValueError, TypeError):
            # If there's any error in parsing the answer
            return 0.0


register_dataset("complex_arithmetic", ComplexArithmeticDataset, ComplexArithmeticConfig)