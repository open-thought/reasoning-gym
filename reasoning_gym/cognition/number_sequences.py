from dataclasses import dataclass
from enum import StrEnum
from random import Random
from typing import List, Optional

from ..factory import ProceduralDataset, register_dataset


class Operation(StrEnum):
    """Basic mathematical operations that can be composed"""

    ADD = "+"
    MULTIPLY = "*"
    SQUARE = "^2"
    DOUBLE = "*2"
    HALF = "/2"
    PREV_PLUS = "prev+"  # Add previous number
    ALTERNATE = "alt"  # Alternate between operations
    COMPOSE = "compose"  # Compose two operations


@dataclass
class NumberSequenceConfig:
    """Configuration for sequence generation"""

    min_terms: int = 4  # Minimum visible terms
    max_terms: int = 8  # Maximum visible terms
    min_value: int = -100  # Minimum allowed number
    max_value: int = 100  # Maximum allowed number
    max_complexity: int = 3  # Maximum number of operations to combine
    seed: Optional[int] = None
    size: int = 500  # Virtual dataset size

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.min_terms >= 4, "need at least 4 terms to establish pattern"
        assert self.max_terms >= self.min_terms
        assert self.max_value > self.min_value
        assert self.max_complexity >= 1


class PatternRule:
    """Represents a composable sequence pattern rule"""

    def __init__(self, operations: List[Operation], parameters: List[int], subrules: List["PatternRule"] = None):
        self.operations = operations
        self.parameters = parameters
        self.subrules = subrules or []

    def apply(self, sequence: List[int], position: int) -> int:
        """Apply the rule to generate the next number"""
        result = sequence[position]  # Start with current number

        for op, param in zip(self.operations, self.parameters):
            if op == Operation.ADD:
                result += param
            elif op == Operation.MULTIPLY:
                result *= param
            elif op == Operation.SQUARE:
                result = result * result
            elif op == Operation.DOUBLE:
                result *= 2
            elif op == Operation.HALF:
                result //= 2  # Integer division
            elif op == Operation.PREV_PLUS:
                if position > 0:
                    result += sequence[position - 1]
            elif op == Operation.COMPOSE:
                # Apply each subrule in sequence, passing the result through
                for subrule in self.subrules:
                    temp_sequence = sequence[: position + 1]
                    temp_sequence[-1] = result  # Use current result as input
                    result = subrule.apply(temp_sequence, position)

        return result

    @classmethod
    def compose(cls, rules: List["PatternRule"]) -> "PatternRule":
        """Create a new rule that composes multiple rules together"""
        return cls([Operation.COMPOSE], [0], subrules=rules)

    def to_string(self) -> str:
        """Convert rule to human-readable string"""
        parts = []
        for op, param in zip(self.operations, self.parameters):
            if op == Operation.ADD:
                parts.append(f"add {param}")
            elif op == Operation.MULTIPLY:
                parts.append(f"multiply by {param}")
            elif op == Operation.SQUARE:
                parts.append("square")
            elif op == Operation.DOUBLE:
                parts.append("double")
            elif op == Operation.HALF:
                parts.append("halve")
            elif op == Operation.PREV_PLUS:
                parts.append("add previous")
        return " then ".join(parts)


class PatternGenerator:
    """Generates new pattern rules with configurable complexity"""

    def __init__(self, rng: Random, complexity: int = 1):
        self.rng = rng
        self.complexity = complexity

    def generate_rule(self) -> PatternRule:
        """Generate a new pattern rule"""
        operations = []
        parameters = []

        # Number of operations based on complexity
        num_ops = self.rng.randint(1, self.complexity + 1)

        for _ in range(num_ops):
            # Pick random operation
            op = self.rng.choice(list(Operation))
            operations.append(op)

            # Generate appropriate parameter
            if op in [Operation.ADD, Operation.MULTIPLY]:
                param = self.rng.randint(-10, 10)
                while param == 0:  # Avoid trivial operations
                    param = self.rng.randint(-10, 10)
                parameters.append(param)
            else:
                parameters.append(0)  # Some operations don't need parameters

        return PatternRule(operations, parameters)

    def is_interesting(self, sequence: List[int], max_value: int = 1000) -> bool:
        """Check if sequence is interesting enough"""
        if not sequence:
            return False

        # Avoid too large numbers
        if any(abs(x) > max_value for x in sequence):
            return False

        # Avoid constant sequences
        if len(set(sequence)) == 1:
            return False

        # Avoid simple arithmetic progressions if complexity > 1
        if self.complexity > 1:
            diffs = [sequence[i + 1] - sequence[i] for i in range(len(sequence) - 1)]
            if len(set(diffs)) == 1:
                return False

        return True


class NumberSequenceDataset(ProceduralDataset):
    """Generates number sequence completion tasks with dynamic pattern generation"""

    def __init__(self, config: NumberSequenceConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def __getitem__(self, idx: int) -> dict:
        """Generate a sequence task with a newly generated pattern"""
        rng = Random(self.seed + idx)

        # Create pattern generator with random complexity
        complexity = rng.randint(1, self.config.max_complexity)
        generator = PatternGenerator(rng, complexity)

        # Generate pattern rule and sequence
        max_attempts = 10
        for _ in range(max_attempts):
            rule = generator.generate_rule()

            # Generate initial terms
            num_terms = rng.randint(self.config.min_terms, self.config.max_terms)
            sequence = [rng.randint(-10, 10)]  # Start with random number

            # Generate remaining terms
            try:
                for i in range(num_terms):  # Generate terms
                    next_term = rule.apply(sequence, i)
                    sequence.append(next_term)

                if generator.is_interesting(sequence):
                    break
            except (OverflowError, ZeroDivisionError):
                continue
        else:
            # If we couldn't generate an interesting sequence, fall back to simple addition
            rule = PatternRule([Operation.ADD], [2])
            sequence = [i * 2 for i in range(num_terms + 1)]

        visible_terms = sequence[:-1]  # Last term is the answer

        return {
            "question": ", ".join(map(str, visible_terms)) + ", ?",
            "answer": str(sequence[-1]),
            "metadata": {"rule": rule.to_string(), "complexity": complexity, "sequence": sequence},
        }


register_dataset("number_sequence", NumberSequenceDataset, NumberSequenceConfig)
