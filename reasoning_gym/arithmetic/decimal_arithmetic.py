from dataclasses import dataclass
from random import Random
from typing import Any, Literal, Optional, Dict

from ..factory import ProceduralDataset, register_dataset


@dataclass
class DecimalArithmeticDatasetConfig:
    """Configuration for decimal arithmetic dataset generation"""

    num_decimal_places: int = 6
    seed: Optional[int] = None
    size: int = 500  # Virtual dataset size

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.num_decimal_places > 0, "num_decimal_places must be positive"


def generate_arithmetic_problem(rng, num_decimal_places, operations=None):
    """
    Generates simple arithmetic problems with decimal numbers formatted to a specific number of decimal places.
    
    Parameters:
        rng
        num_problems (int): Number of problems to generate
        num_decimal_places (int): Number of decimal places for the numbers
        operations (list): List of operations to use (default: ['+', '-', '*', '/'])
    
    Returns:
        list: List of formatted arithmetic problem strings
    """
    if operations is None:
        operations = ['+', '-', '*', '/']
    
    max_integer_part = 10  # Maximum whole number portion before decimal
    max_value = max_integer_part * (10 ** num_decimal_places)
    
    problem = None

    # Generate random numbers with exact decimal places
    num1 = rng.randint(1, max_value) / (10 ** num_decimal_places)
    num2 = rng.randint(1, max_value) / (10 ** num_decimal_places)
    
    # Select random operation
    op = rng.choice(operations)
    
    # Format numbers to ensure exact decimal places
    formatted_num1 = f"{num1:.{num_decimal_places}f}"
    formatted_num2 = f"{num2:.{num_decimal_places}f}"
    
    problem = f"{formatted_num1} {op} {formatted_num2} = ?"
    
    return problem


def eval_floordiv(exp: str) -> int:
    return eval(exp.replace("/", "//").replace(" = ?", ''))


class DecimalArithmeticDataset(ProceduralDataset):
    """Dataset that generates basic arithmetic tasks with configurable complexity"""

    def __init__(self, config: DecimalArithmeticDatasetConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

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
        rng = Random(self.seed + idx)

        decimal_problem = generate_arithmetic_problem(rng, self.config.num_decimal_places)
        answer = eval_floordiv(decimal_problem)

        return {
            "question": decimal_problem,
            "answer": answer,
            "metadata": {
                
            }
        }

    def score_answer(self, answer: Optional[str], entry: Dict[str, any]) -> float:
        """Determine if the solution provided solves the Sokoban task.

        The function awards 1.0 for a correct answer.

        Args:
            answer (Optional[str]): The user's answer.
            entry (Dict[str, any]): The original dataset entry containing the correct answer.

        Returns:
            float: The computed score between 0.0 and 1.0.
        """

        if answer == None:
            return 0.0

        try:
            if float(answer) == entry['answer']:
                return 1.0
        except Exception as e:
            return 0.01

        return 0.01

# Register the dataset
register_dataset("decimal_arithmetic", DecimalArithmeticDataset, DecimalArithmeticDatasetConfig)
