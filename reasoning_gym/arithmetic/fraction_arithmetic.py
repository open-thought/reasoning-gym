from dataclasses import dataclass
import random
from fractions import Fraction
from typing import Any, Dict, Optional
from ..factory import ProceduralDataset

@dataclass
class FractionArithmeticConfig:
    min_denominator: int = 2
    max_denominator: int = 12
    min_terms: int = 2
    max_terms: int = 3
    allow_subtraction: bool = True
    seed: Optional[int] = None
    size: int = 100

class FractionArithmeticDataset(ProceduralDataset):
    def __init__(self, config: FractionArithmeticConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)
        self.config = config

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rng = random.Random(self.seed + idx)
        
        terms = []
        ops = []
        for i in range(rng.randint(self.config.min_terms, self.config.max_terms)):
            den = rng.randint(self.config.min_denominator, self.config.max_denominator)
            num = rng.randint(1, den * 2)
            terms.append(Fraction(num, den))
            if i > 0:
                op_choices = ['+', '-'] if self.config.allow_subtraction else ['+']
                ops.append(rng.choice(op_choices))

        result = terms[0]
        expr_str = f"{terms[0].numerator}/{terms[0].denominator}"
        for op, term in zip(ops, terms[1:]):
            expr_str += f" {op} {term.numerator}/{term.denominator}"
            if op == '+':
                result += term
            elif op == '-':
                result -= term

        answer = f"{result.numerator}/{result.denominator}"

        return {
            "question": f"Evaluate the following fraction expression and simplify your answer as numerator/denominator (e.g., 3/4): {expr_str}",
            "answer": answer,
            "metadata": {
                "expression": expr_str,
                "result": answer,
            }
        }