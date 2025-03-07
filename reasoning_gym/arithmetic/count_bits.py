"""Count number of 1 bits in a number."""

from dataclasses import dataclass
from random import Random
from typing import Optional

from ..coaching import AttributeType, BaseCurriculum, RangeAttributeDefinition
from ..factory import ProceduralDataset, register_dataset

QUESTION_TEMPLATE = """How many 1 bits are there in the binary representation of the number {number}?"""


@dataclass
class CountBitsConfig:
    """Configuration for Count Bits dataset generation"""

    min_n: int = 1  # Minimum number to consider
    max_n: int = 2**31 - 1  # Maximum number to consider

    size: int = 500  # Virtual dataset size
    seed: Optional[int] = None

    def validate(self):
        """Validate configuration parameters"""
        assert 1 <= self.min_n <= self.max_n, "min_n must be between 1 and max_n"


class CountBitsDataset(ProceduralDataset):
    """Generates Count Bits exercises with configurable difficulty"""

    def __init__(self, config: CountBitsConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def __getitem__(self, idx: int) -> dict:
        """Generate a single Count Bits question"""
        rng = Random(self.seed + idx)

        number = rng.randint(self.config.min_n, self.config.max_n)
        binary = bin(number)[2:]
        answer = binary.count("1")

        return {
            "question": QUESTION_TEMPLATE.format(number=number),
            "answer": str(answer),
            "metadata": {
                "number": number,
                "solution": answer,
                "binary": binary,
                "difficulty": {"n": number},
            },
        }


class CountBitsCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(CountBitsCurriculum.__name__, CountBitsConfig)

        # Define attributes
        self._define_attributes(
            RangeAttributeDefinition(
                name="n",
                levels=[1_000, 1_000_000, 100_000_000, 2**31 - 1],
                default_level=0,
                description="Number to count bits in",
                attr_type=AttributeType.APPEND,
                min_value=1,
                lower_field_name="min_n",
                upper_field_name="max_n",
            ),
        )


register_dataset("count_bits", CountBitsDataset, CountBitsConfig, CountBitsCurriculum)
