"""Check if two strings are isomorphic.

Two strings are isomorphic if the characters in one string can be replaced to get the second string.

A popular Leetcode problem:
https://leetcode.com/problems/isomorphic-strings/description/
"""

from dataclasses import dataclass
from random import Random
from typing import Optional

from ..factory import ProceduralDataset, register_dataset

QUESTION_TEMPLATE = """Two strings are isomorphic if the characters in one string can be replaced to get the second string.

All occurrences of a character must be replaced with another character while preserving the order of characters.

No two characters may map to the same character, but a character may map to itself.

Example 1:
Input: egg add
Output: True
Explanation: The strings s and t can be made identical by:
    - Mapping 'e' to 'a'.
    - Mapping 'g' to 'd'.

Example 2:
Input: foo bar
Output: False
Explanation:
    - The strings cannot be made identical as 'o' needs to be mapped to both 'a' and 'r'.

Return True if the following two strings are isomorphic, or False otherwise:
{s} {t}
"""


@dataclass
class IsomorphicStringsConfig:
    """Configuration for Isomorphic Strings dataset generation"""

    max_string_length: int = 10  # Maximum length of the strings
    p_solvable: float = 0.5  # Probability that the generated question is solvable

    size: int = 500  # Virtual dataset size
    seed: Optional[int] = None

    def validate(self):
        """Validate configuration parameters"""
        assert 2 <= self.max_string_length, "max_string_length must be at least 2"
        assert 0 <= self.p_solvable <= 1, "p_solvable must be between 0 and 1"


class IsomorphicStringsDataset(ProceduralDataset):
    """Generates Isomorphic Strings exercises with configurable difficulty"""

    def __init__(self, config: IsomorphicStringsConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)
        self.letters = {chr(i) for i in range(ord("a"), ord("z") + 1)}

    def _check_isomorphic(self, s: str, t: str) -> bool:
        """Check if two strings are isomorphic"""
        if len(s) != len(t):
            return False

        mapping, inverse_mapping = {}, {}  # s -> t, t -> s
        for i in range(len(s)):
            if (s[i] in mapping and mapping[s[i]] != t[i]) or (
                t[i] in inverse_mapping and s[i] != inverse_mapping[t[i]]
            ):
                return False
            mapping[s[i]] = t[i]
            inverse_mapping[t[i]] = s[i]

        return True

    def _generate_inputs(self, rng: Random, solvable: bool) -> tuple[str, str]:
        """Generate the two input strings"""
        s, t = [], []
        mapping = {}

        # Generate a valid isomorphic pair first (leave one character for potential conflict)
        for _ in range(rng.randint(1, self.config.max_string_length - 1)):
            char_s = rng.choice(list(self.letters))
            if char_s not in mapping:
                # Choose a random character that is not already mapped
                char_t = rng.choice(list(self.letters - set(mapping.values())))
                mapping[char_s] = char_t
            else:
                # Use the existing mapping
                char_t = mapping[char_s]
            s.append(char_s)
            t.append(char_t)

        if not solvable:
            # Solution should be unsolvable, create conflict
            letter = rng.choice(list(mapping.keys()))
            conflict = rng.choice(list(self.letters - {mapping[letter]}))
            insert_idx = rng.randint(0, len(s))
            s.insert(insert_idx, letter)
            t.insert(insert_idx, conflict)

        return "".join(s), "".join(t)

    def __getitem__(self, idx: int) -> dict:
        """Generate a single Isomorphic Strings question"""
        rng = Random(self.seed + idx)

        solvable = rng.random() < self.config.p_solvable
        s, t = self._generate_inputs(rng, solvable)
        answer = self._check_isomorphic(s, t)

        return {
            "question": QUESTION_TEMPLATE.format(s=s, t=t),
            "answer": str(answer),
            "metadata": {"words": [s, t], "solution": answer, "solvable": solvable},
        }


register_dataset("isomorphic_strings", IsomorphicStringsDataset, IsomorphicStringsConfig)
