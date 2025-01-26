"""Fraction simplification task generator"""

from dataclasses import dataclass
from math import gcd
from random import Random
from typing import Optional, Sequence, Tuple

from ..factory import ProceduralDataset, register_dataset


@dataclass
class FractionSimplificationConfig:
    """Configuration for fraction simplification task generation"""

    min_value: int = 1  # Minimum value for numerator/denominator
    max_value: int = 1000  # Maximum value for numerator/denominator
    min_factor: int = 1  # Minimum multiplication factor
    max_factor: int = 100  # Maximum multiplication factor
    styles: Sequence[str] = ("plain", "latex_inline", "latex_frac", "latex_dfrac")  # Allowed fraction formatting styles
    seed: Optional[int] = None
    size: int = 500  # Virtual dataset size

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.min_value > 0, "min_value must be positive"
        assert self.max_value > self.min_value, "max_value must be > min_value"
        assert self.min_factor >= 1, "min_factor must be at least 1"
        assert self.max_factor >= self.min_factor, "max_factor must be >= min_factor"

        # Validate styles
        valid_styles = {"plain", "latex_inline", "latex_frac", "latex_dfrac"}
        for style in self.styles:
            assert style in valid_styles, f"Invalid style: {style}. Must be one of {valid_styles}"


class FractionSimplificationDataset(ProceduralDataset):
    """Generates fraction simplification tasks"""

    def __init__(self, config: FractionSimplificationConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _generate_fraction(self, rng: Random) -> Tuple[int, int, int, int]:
        """Generate a random fraction and its simplified form.
        Returns (numerator, denominator, simplified_num, simplified_den)"""
        # Try to generate valid fractions until we get one that meets our criteria
        for _ in range(10):  # Limit attempts to avoid infinite loop
            # Generate the simplified fraction first
            simplified_num = rng.randint(self.config.min_value, self.config.max_value)
            simplified_den = rng.randint(self.config.min_value, self.config.max_value)

            # Make sure they're coprime by dividing by their GCD
            common = gcd(simplified_num, simplified_den)
            simplified_num //= common
            simplified_den //= common

            # Check if simplified fraction is within bounds
            if (
                self.config.min_value <= simplified_num <= self.config.max_value
                and self.config.min_value <= simplified_den <= self.config.max_value
            ):
                # Ensure numerator is smaller than denominator
                if simplified_num > simplified_den:
                    simplified_num, simplified_den = simplified_den, simplified_num

                # Multiply both by a random factor to create the unsimplified version
                factor = rng.randint(self.config.min_factor, self.config.max_factor)
                numerator = simplified_num * factor
                denominator = simplified_den * factor
                return numerator, denominator, simplified_num, simplified_den

        # If we failed to find a good fraction after max attempts,
        # generate one that's guaranteed to be within bounds
        simplified_num = rng.randint(self.config.min_value, self.config.max_value)
        simplified_den = rng.randint(self.config.min_value, self.config.max_value)

        # Ensure numerator is smaller than denominator
        if simplified_num > simplified_den:
            simplified_num, simplified_den = simplified_den, simplified_num

        factor = rng.randint(self.config.min_factor, self.config.max_factor)
        return (simplified_num * factor, simplified_den * factor, simplified_num, simplified_den)

    def _format_fraction(self, num: int, den: int, style: str = "plain") -> str:
        """Format a fraction in various styles"""
        if style == "plain":
            return f"{num}/{den}"
        elif style == "latex_inline":
            return f"${num}/{den}$"
        elif style == "latex_frac":
            return f"$\\frac{{{num}}}{{{den}}}$"
        elif style == "latex_dfrac":
            return f"$\\dfrac{{{num}}}{{{den}}}$"
        else:
            raise ValueError(f"Unknown fraction style: {style}")

    def __getitem__(self, idx: int) -> dict:
        """Generate a single fraction simplification task"""
        rng = Random(self.seed + idx)

        num, den, simple_num, simple_den = self._generate_fraction(rng)

        # Choose a random style from configured styles
        style = self.config.styles[rng.randint(0, len(self.config.styles) - 1)]

        # Format both question and answer in the same style
        question_fraction = self._format_fraction(num, den, style)
        answer_fraction = self._format_fraction(simple_num, simple_den, style)

        return {
            "question": f"Simplify the fraction {question_fraction} to its lowest terms",
            "answer": answer_fraction,
            "metadata": {
                "numerator": num,
                "denominator": den,
                "simplified_numerator": simple_num,
                "simplified_denominator": simple_den,
                "reduction_factor": num // simple_num,  # Will be same as den // simple_den
                "style": style,
            },
        }


register_dataset("fraction_simplification", FractionSimplificationDataset, FractionSimplificationConfig)
