from dataclasses import dataclass
import random
from typing import Optional


@dataclass
class ChainSumConfig:
    """Configuration for chain sum task generation"""
    min_terms: int = 2
    max_terms: int = 6
    min_digits: int = 1
    max_digits: int = 4
    allow_negation: bool = False
    seed: Optional[int] = None
    size: int = 500

    def validate(self):
        """Validate configuration parameters"""
        assert self.min_terms > 0, "min_terms must be positive"
        assert self.max_terms >= self.min_terms, "max_terms must be >= min_terms"
        assert self.min_digits > 0, "min_digits must be positive"
        assert self.max_digits >= self.min_digits, "max_digits must be >= min_digits"
        
        # Validate digit ranges make sense
        if self.min_digits > 1:
            assert 10 ** (self.min_digits - 1) >= 1, "min_digits would result in invalid number range"


class ChainSum:
    """Generates simple arithmetic tasks using only + and - operators"""
    
    def __init__(self, config: ChainSumConfig):
        self.config = config
        self.config.validate()
        # Generate base seed if none provided
        self.seed = config.seed if config.seed is not None else random.randint(0, 2**32)
        
    def __len__(self) -> int:
        return self.config.size
        
    def __getitem__(self, idx: int) -> dict:
        """Generate a single chain sum task
        
        Args:
            idx: Index of the item to generate
            
        Returns:
            dict with keys:
                - question: str, the formatted arithmetic expression
                - answer: str, the ground truth result
                - metadata: dict with generation parameters
        """
        # Create deterministic RNG from base seed and idx
        item_rng = random.Random(self.seed + idx)
        
        num_terms = item_rng.randint(self.config.min_terms, self.config.max_terms)
        num_digits = item_rng.randint(self.config.min_digits, self.config.max_digits)
        
        expression, result = self._generate_task(item_rng, num_terms, num_digits)
            
        return {
            "question": f"{expression} =",
            "answer": str(result),
            "metadata": {
                "num_terms": num_terms,
                "num_digits": num_digits,
                "expression": expression
            }
        }

    def _generate_task(self, rng: random.Random, num_terms: int, num_digits: int) -> tuple[str, int]:
        """Generate a chain sum task
        
        Args:
            rng: Random number generator
            num_terms: Number of terms in the expression
            num_digits: Number of digits for each number
            
        Returns:
            Tuple of (expression string, result integer)
        """
        # Generate numbers with at least min_digits
        min_value = 10 ** (num_digits - 1)  # e.g., 100 for 3 digits
        max_value = (10 ** num_digits) - 1   # e.g., 999 for 3 digits
        
        if self.config.allow_negation:
            # Allow both positive and negative numbers in the range
            constants = [rng.randint(-max_value, max_value) for _ in range(num_terms)]
        else:
            # Only positive numbers
            constants = [rng.randint(min_value, max_value) for _ in range(num_terms)]
        operators = [rng.choice(["+", "-"]) for _ in range(num_terms - 1)]

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
            else:  # op == "-"
                result -= c

        expression = " ".join(expression_parts)
        return expression, result
