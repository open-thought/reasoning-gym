import random
from dataclasses import dataclass
from typing import Optional, Tuple
import string

import sympy
from sympy import Symbol, solve, Eq

from ..dataset import ProceduralDataset


@dataclass
class SimpleEquationsConfig:
    """Configuration for simple equation task generation"""
    min_terms: int = 2  # Minimum number of terms in expression
    max_terms: int = 4  # Maximum number of terms
    min_value: int = 1  # Minimum value for constants
    max_value: int = 20  # Maximum value for constants
    operators: tuple = ('+', '-', '*')  # Allowed operators
    seed: Optional[int] = None
    size: int = 500

    def validate(self):
        """Validate configuration parameters"""
        assert self.min_terms > 0, "min_terms must be positive"
        assert self.max_terms >= self.min_terms, "max_terms must be >= min_terms"
        assert self.min_value > 0, "min_value must be positive"
        assert self.max_value >= self.min_value, "max_value must be >= min_value"
        assert len(self.operators) > 0, "must specify at least one operator"


class SimpleEquationsDataset(ProceduralDataset):
    """Generates simple equations with one variable to solve"""

    def __init__(self, config: SimpleEquationsConfig):
        self.config = config
        self.config.validate()
        super().__init__(seed=config.seed, size=config.size)

    def __getitem__(self, idx: int) -> dict:
        """Generate a single equation task
        
        Returns:
            dict with keys:
                - question: str, the equation to solve (e.g. "3 * x = 12")
                - answer: str, the solution value (e.g. "4")
                - metadata: dict with generation parameters
        """
        rng = random.Random(self.seed + idx)
        
        # Get variable and generate equation
        variable = self._get_variable(rng)
        equation, solution = self._generate_equation(rng, variable)
        
        return {
            "question": equation,
            "answer": str(solution),
            "metadata": {
                "equation": equation,
                "variable": variable,
            }
        }

    def _get_variable(self, rng: random.Random) -> str:
        """Get a random lowercase variable name"""
        return rng.choice(string.ascii_lowercase)

    def _generate_equation(self, rng: random.Random, variable: str) -> Tuple[str, int]:
        """Generate an equation and its solution
        
        Args:
            rng: Random number generator
            variable: Variable symbol to use in equation
            
        Returns:
            Tuple of (equation string, solution integer)
        """
        x = Symbol(variable)
        
        # Generate left side
        num_terms = rng.randint(self.config.min_terms, self.config.max_terms)
        terms = []
        
        # First term includes the variable
        coef = rng.randint(self.config.min_value, self.config.max_value)
        terms.append(coef * x)
        
        # Add remaining terms
        for _ in range(num_terms - 1):
            value = rng.randint(self.config.min_value, self.config.max_value)
            op = rng.choice(self.config.operators)
            
            if op == '+':
                terms.append(value)
            elif op == '-':
                terms.append(-value)
            else:  # '*'
                terms[-1] = terms[-1] * value
        
        left_side = sum(terms)
        
        # Generate right side
        right_side = rng.randint(self.config.min_value, self.config.max_value)
        
        # Create equation
        equation = Eq(left_side, right_side)
        solution = solve(equation, x)[0]
        
        # Only return if solution is a positive integer
        if not (isinstance(solution, sympy.Integer) and solution > 0):
            return self._generate_equation(rng, variable)
            
        return f"{left_side} = {right_side}", int(solution)


def simple_equations_dataset(
    min_terms: int = 2,
    max_terms: int = 4,
    min_value: int = 1,
    max_value: int = 20,
    operators: tuple = ('+', '-', '*'),
    seed: Optional[int] = None,
    size: int = 500,
) -> SimpleEquationsDataset:
    """Create a SimpleEquationsDataset with the given configuration"""
    config = SimpleEquationsConfig(
        min_terms=min_terms,
        max_terms=max_terms,
        min_value=min_value,
        max_value=max_value,
        operators=operators,
        seed=seed,
        size=size,
    )
    return SimpleEquationsDataset(config)
