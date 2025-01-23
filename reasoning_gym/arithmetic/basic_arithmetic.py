from dataclasses import dataclass
from random import Random
from typing import Optional, Literal, Any


@dataclass
class ArithmeticDatasetConfig:
    """Configuration for arithmetic dataset generation"""
    min_terms: int = 2
    max_terms: int = 6
    min_digits: int = 1
    max_digits: int = 4
    operators: list[str] = ("+" , "-", "*")
    allow_parentheses: bool = True
    allow_negation: bool = True
    seed: Optional[int] = None
    size: int = 500  # Virtual dataset size
    format_style: Literal["simple", "natural"] = "simple"

    def validate(self):
        """Validate configuration parameters"""
        assert self.min_terms > 0, "min_terms must be positive"
        assert self.max_terms >= self.min_terms, "max_terms must be >= min_terms"
        assert self.min_digits > 0, "min_digits must be positive"
        assert self.max_digits >= self.min_digits, "max_digits must be >= min_digits"
        assert len(self.operators) > 0, "must provide at least one operator"
        for op in self.operators:
            assert op in ["+", "-", "*"], f"unsupported operator: {op}"


class ArithmeticDataset:
    """Dataset that generates arithmetic tasks with configurable complexity"""
    
    def __init__(self, config: ArithmeticDatasetConfig):
        self.config = config
        self.config.validate()
        self.rng = Random(config.seed)
        
    def __len__(self) -> int:
        return self.config.size
        
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
        # Use seed derived from idx for deterministic generation
        item_rng = Random(self.rng.randint(0, 2**32) + idx)
        
        num_terms = item_rng.randint(self.config.min_terms, self.config.max_terms)
        num_digits = item_rng.randint(self.config.min_digits, self.config.max_digits)
        
        if self.config.allow_parentheses:
            expression, result = self._generate_complex_task(item_rng, num_terms, num_digits)
        else:
            expression, result = self._generate_simple_task(item_rng, num_terms, num_digits)
            
        question = self._format_question(expression)
            
        return {
            "question": question,
            "answer": str(result),
            "metadata": {
                "num_terms": num_terms,
                "num_digits": num_digits,
                "expression": expression
            }
        }

    def _generate_complex_task(self, rng: Random, num_terms: int, num_digits: int) -> tuple[str, int]:
        """Generate a complex arithmetic task with possible parentheses"""
        parts = []

        def add_terms(remaining: int):
            num_left = rng.randint(1, remaining)
            num_right = remaining - num_left

            if num_left > 1 and rng.random() > 0.5 and self.config.allow_parentheses:
                if rng.random() > 0.5 and self.config.allow_negation:
                    parts.append("-(")
                else:
                    parts.append("(")
                add_terms(num_left)
                parts.append(")")
            else:
                for i in range(num_left):
                    c = rng.randint(-(10**num_digits) + 1, 10**num_digits - 1)
                    parts.append(str(c))
                    if i + 1 < num_left:
                        parts.append(rng.choice(self.config.operators))

            if num_right > 0:
                parts.append(rng.choice(self.config.operators))
                add_terms(num_right)

        add_terms(num_terms)

        # Add at most one random space between parts
        space_parts = []
        for p in parts:
            if rng.random() < 0.15:
                space_parts.append(" ")
            space_parts.append(p)

        expression = " ".join(space_parts).strip()
        result = eval(expression)  # Note: eval is safe here as we control the input

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
            else:
                raise RuntimeError(f"Unsupported operator: {op}")

        expression = " ".join(expression_parts)
        return expression, result

    def __iter__(self):
        """Make the dataset iterable"""
        self._current_idx = 0
        return self
        
    def __next__(self):
        """Get next item in iteration"""
        if self._current_idx >= self.config.size:
            raise StopIteration
        item = self[self._current_idx]
        self._current_idx += 1
        return item

    def _format_question(self, expression: str) -> str:
        """Format the expression according to config style"""
        if self.config.format_style == "simple":
            return f"{expression} ="
        else:
            templates = [
                "What is {0}?",
                "Calculate {0}",
                "Solve {0}",
                "Evaluate the expression: {0}"
            ]
            return self.rng.choice(templates).format(expression)
