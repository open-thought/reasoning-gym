"""Base conversion task generator"""

from dataclasses import dataclass
from random import Random
from typing import Optional, Tuple

from ..factory import ProceduralDataset, register_dataset


@dataclass
class BaseConversionConfig:
    """Configuration for base conversion task generation"""

    min_base: int = 2  # Minimum base (2=binary)
    max_base: int = 16  # Maximum base (16=hex)
    min_value: int = 0  # Minimum decimal value to convert
    max_value: int = 1000  # Maximum decimal value to convert
    seed: Optional[int] = None
    size: int = 500  # Virtual dataset size

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert 2 <= self.min_base <= 36, "min_base must be between 2 and 36"
        assert self.min_base <= self.max_base <= 36, "max_base must be between min_base and 36"
        assert self.min_value >= 0, "min_value must be non-negative"
        assert self.max_value > self.min_value, "max_value must be > min_value"


class BaseConversionDataset(ProceduralDataset):
    """Generates base conversion tasks"""

    def __init__(self, config: BaseConversionConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _format_base_name(self, base: int) -> str:
        """Get human-readable name for common bases"""
        if base == 2:
            return "binary"
        elif base == 16:
            return "hexadecimal"
        else:
            return f"base-{base}"

    def _generate_conversion(self, rng: Random) -> Tuple[int, int, int]:
        """Generate random value and source/target bases"""
        value = rng.randint(self.config.min_value, self.config.max_value)

        # Choose source and target bases
        source_base = rng.randint(self.config.min_base, self.config.max_base)
        target_base = rng.randint(self.config.min_base, self.config.max_base)
        while target_base == source_base:  # Ensure different bases
            target_base = rng.randint(self.config.min_base, self.config.max_base)

        return value, source_base, target_base

    def __getitem__(self, idx: int) -> dict:
        """Generate a single base conversion task"""
        rng = Random(self.seed + idx)

        value, source_base, target_base = self._generate_conversion(rng)

        # Convert decimal to source base representation
        if source_base == 16:
            source_repr = format(value, "x")
        elif source_base == 2:
            source_repr = format(value, "b")
        else:
            # Manual conversion for other bases
            n = value
            digits = []
            while n:
                digits.append(int(n % source_base))
                n //= source_base
            source_repr = "".join(str(d) if d < 10 else chr(ord("a") + d - 10) for d in reversed(digits) or [0])

        # Convert decimal to target base for answer
        if target_base == 16:
            target_repr = format(value, "x")
        elif target_base == 2:
            target_repr = format(value, "b")
        else:
            # Manual conversion for other bases
            n = value
            digits = []
            while n:
                digits.append(int(n % target_base))
                n //= target_base
            target_repr = "".join(str(d) if d < 10 else chr(ord("a") + d - 10) for d in reversed(digits) or [0])

        source_name = self._format_base_name(source_base)
        target_name = self._format_base_name(target_base)

        # Add hint for bases > 10 about using lowercase letters
        hint = " (use lowercase letters a-z for digits above 9)" if target_base > 10 else ""

        return {
            "question": f"Convert the {source_name} number {source_repr} to {target_name}{hint}",
            "answer": target_repr,
            "metadata": {
                "decimal_value": value,
                "source_base": source_base,
                "target_base": target_base,
                "source_repr": source_repr,
                "target_repr": target_repr,
            },
        }


register_dataset("base_conversion", BaseConversionDataset, BaseConversionConfig)
