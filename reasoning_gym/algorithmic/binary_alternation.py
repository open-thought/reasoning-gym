"""Minimum number of swaps to make a binary string alternating

https://leetcode.com/problems/minimum-number-of-swaps-to-make-the-binary-string-alternating/description/
"""

from dataclasses import dataclass
from random import Random
from typing import Optional

from ..coaching import AttributeType, BaseCurriculum, RangeAttributeDefinition
from ..factory import ProceduralDataset, register_dataset

QUESTION_TEMPLATE = """Given a binary string, return the minimum number of character swaps to make it alternating, or -1 if it is impossible.

The string is called alternating if no two adjacent characters are equal. For example, the strings "010" and "1010" are alternating, while the string "0100" is not.

Any two characters may be swapped, even if they are not adjacent.

Now, determine the minimum number of swaps to make the following binary string alternating: {string}
"""


@dataclass
class BinaryAlternationConfig:
    """Configuration for Count Bits dataset generation"""

    min_n: int = 10  # Minimum number of bits in the binary string
    max_n: int = 30  # Maximum number of bits in the binary string
    p_solvable: float = 0.8  # Probability of generating a solvable sample

    size: int = 500  # Virtual dataset size
    seed: Optional[int] = None

    def validate(self):
        """Validate configuration parameters"""
        assert 1 <= self.min_n, "Minimum number of bits must be at least 1"
        assert self.min_n <= self.max_n, "Minimum number of bits must be <= maximum number of bits"
        assert 0 <= self.p_solvable <= 1, "Probability of generating a 1 must be in [0, 1]"


class BinaryAlternationDataset(ProceduralDataset):
    """Generates Binary Alternation exercises with configurable difficulty"""

    def __init__(self, config: BinaryAlternationConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _get_binary_string(self, rng: Random, n: int, solvable: bool) -> str:
        ones, zeros = n // 2, n // 2

        # Check if we need to add an extra bit
        if n % 2 == 1:
            if rng.random() < 0.5:
                ones += 1
            else:
                zeros += 1

        if not solvable:
            if ones > zeros:
                ones += 1
            elif ones < zeros:
                zeros += 1
            else:
                # Randomly add 2 bits of the same type
                if rng.random() < 0.5:
                    ones += 2
                else:
                    zeros += 2

        # Generate the string
        string = ["1"] * ones + ["0"] * zeros
        rng.shuffle(string)
        return "".join(string)

    def _get_answer(self, string: str) -> int:
        """Calculate the minimum number of swaps to make the string alternating"""

        def get_num_swaps(expected):
            incorrect = 0
            for c in string:
                if c != expected:
                    incorrect += 1
                expected = "1" if expected == "0" else "0"
            return incorrect // 2  # number of swaps is half of incorrect positions

        ones, zeros = string.count("1"), string.count("0")
        if abs(ones - zeros) > 1:
            return -1  # impossible to make alternating
        if ones > zeros:
            return get_num_swaps("1")
        elif ones < zeros:
            return get_num_swaps("0")
        else:
            return min(get_num_swaps("0"), get_num_swaps("1"))

    def __getitem__(self, idx: int) -> dict:
        """Generate a single Count Bits question"""
        rng = Random(self.seed + idx)

        n = rng.randint(self.config.min_n, self.config.max_n)
        solvable = rng.random() < self.config.p_solvable
        string = self._get_binary_string(rng, n, solvable)
        answer = self._get_answer(string)

        return {
            "question": QUESTION_TEMPLATE.format(string=string),
            "answer": str(answer),
            "metadata": {
                "string": string,
                "solution": answer,
                "solvable": solvable,
                "difficulty": {"n": n},
            },
        }


class BinaryAlternationCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(BinaryAlternationCurriculum.__name__, BinaryAlternationConfig)

        # Define attributes
        self._define_attributes(
            RangeAttributeDefinition(
                name="n",
                levels=[10, 50, 500, 1000],
                default_level=0,
                description="Number of bits in the binary string",
                attr_type=AttributeType.APPEND,
                min_value=1,
                lower_field_name="min_n",
                upper_field_name="max_n",
            )
        )


register_dataset("binary_alternation", BinaryAlternationDataset, BinaryAlternationConfig, BinaryAlternationCurriculum)
