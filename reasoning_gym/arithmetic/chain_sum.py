from dataclasses import dataclass
from typing import Dict, Any, List
import random
from reasoning_gym.core.base_curriculum import BaseCurriculum, Template, Placeholder

@dataclass
class SymbolicExpression:
    """Represents a symbolic mathematical expression with lazy evaluation"""
    template: str
    placeholders: Dict[str, 'SymbolicTerm']
    metadata: Dict[str, Any]

@dataclass
class SymbolicTerm:
    """Represents a symbolic term that can be evaluated at runtime"""
    sign_gen: Any      # Generator from sign attribute
    notation_gen: Any  # Generator from notation attribute
    base_gen: Any      # Generator from base attribute
    num_gen: Any       # Composite generator for number (uses digits/decimals)

    def evaluate(self, rng: random.Random) -> Dict[str, Any]:
        sign = self.sign_gen()
        notation = self.notation_gen()
        base = self.base_gen()
        number = self.num_gen()

        # Format based on notation and base
        match notation:
            case "regular":
                text = str(number)
            case "scientific":
                text = f"{number:.2e}"
            case "base":
                match base:
                    case 2:
                        text = f"0b{bin(number)[2:]}"
                    case 16:
                        text = f"0x{hex(number)[2:].upper()}"
                    case _:
                        text = str(number)

        return {
            "text": f"{sign}{text}",
            "value": number if sign != '-' else -number,
            "metadata": {
                "notation": notation,
                "base": base,
                "raw_value": number,
                "sign": sign
            }
        }

class ChainSumDataset:
    def __init__(self):
        self.rng = random.Random()

    def generate_expression(self, curriculum: BaseCurriculum) -> Dict[str, Any]:
        """Generates a symbolic expression based on current curriculum levels"""
        # Get generators for each attribute
        num_terms = curriculum.get_attr_value("max_terms")
        operators = curriculum.get_attr_value("operators")

        # Create symbolic terms
        terms: List[SymbolicTerm] = []
        for i in range(num_terms):
            term = SymbolicTerm(
                sign_gen=curriculum.attributes["sign"].get_generator(
                    curriculum.get_attr_level("sign"), self.rng),
                notation_gen=curriculum.attributes["notation"].get_generator(
                    curriculum.get_attr_level("notation"), self.rng),
                base_gen=curriculum.attributes["base"].get_generator(
                    curriculum.get_attr_level("base"), self.rng),
                num_gen=self._create_number_generator(curriculum)
            )
            terms.append(term)

        # Create symbolic expression template
        placeholders = {}
        template_parts = []

        for i in range(num_terms):
            # Add term
            term_key = f"term_{i}"
            template_parts.append(f"{{{term_key}}}")
            placeholders[term_key] = terms[i]

            # Add operator if not last term
            if i < num_terms - 1:
                op_key = f"op_{i}"
                template_parts.append(f"{{{op_key}}}")
                placeholders[op_key] = lambda: self.rng.choice(operators)

        return SymbolicExpression(
            template=" ".join(template_parts),
            placeholders=placeholders,
            metadata={
                "num_terms": num_terms,
                "available_operators": operators
            }
        )

    def _create_number_generator(self, curriculum: BaseCurriculum):
        """Creates a composite generator for numbers based on digits and decimals"""
        digits_gen = curriculum.attributes["num_digits"].get_generator(
            curriculum.get_attr_level("num_digits"), self.rng)
        decimals_gen = curriculum.attributes["num_decimals"].get_generator(
            curriculum.get_attr_level("num_decimals"), self.rng)

        def generate_number():
            max_digits = digits_gen()
            decimals = decimals_gen()

            # Generate integer part
            number = self.rng.randint(10 ** (max_digits-1), 10 ** max_digits - 1)

            # Add decimal places if needed
            if decimals > 0:
                decimal_part = self.rng.randint(0, 10 ** decimals - 1)
                return number + (decimal_part / (10 ** decimals))
            return number

        return generate_number

    def generate(self) -> Dict[str, Any]:
        """Main generation entry point"""
        symbolic = self.generate_expression(self.curriculum)

        # Evaluate all placeholders
        evaluated_placeholders = {}
        raw_values = []
        operators = []

        for key, placeholder in symbolic.placeholders.items():
            if key.startswith('term_'):
                result = placeholder.evaluate(self.rng)
                evaluated_placeholders[key] = result["text"]
                raw_values.append(result["value"])
            else:  # operator
                op = placeholder()
                evaluated_placeholders[key] = op
                operators.append(op)

        # Format final expression
        expression = symbolic.template.format(**evaluated_placeholders)

        # Calculate final answer
        answer = raw_values[0]
        for i, op in enumerate(operators):
            match op:
                case '+': answer += raw_values[i+1]
                case '-': answer -= raw_values[i+1]
                case '*': answer *= raw_values[i+1]
                case '/': answer /= raw_values[i+1]
                case '**': answer **= raw_values[i+1]

        return {
            "question": f"Calculate the following: {expression}",
            "answer": str(answer),
            "metadata": {
                "raw_values": raw_values,
                "operators": operators,
                "expression": expression,
                **symbolic.metadata
            }
        }
