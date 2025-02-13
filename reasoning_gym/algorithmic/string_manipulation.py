"""Manipulate a string according to a set of rules

https://github.com/yongchao98/CodeSteer-v1.0/blob/main/create_dataset/create_dataset_string_deletion_and_modification.py
"""

from dataclasses import dataclass
from random import Random
from typing import Optional

from ..factory import ProceduralDataset, register_dataset

QUESTION_TEMPLATE = """Your job is to repeatedly transform a string according to a set of rules until no further transformations can be performed, or a state is repeated.

Evaluate the following rules in order, and apply the first applicable rule to the string:
{rules}

Once you have applied a rule, repeat the process with the new string until no further transformations can be performed (i.e. the string doesn't change), or a state is repeated.
If a state is repeated, the process is terminated, and the repeated state is discarded (i.e. is not considered as the final answer) and the state before the repeated state is considered as the final answer.

Example:
- Input:
    - String: abbac
    - Rules:
        1. If the string prefix is 'ab', replace it with 'ca'.
        2. If the string prefix is 'ca', replace it with 'bb' and append 'c' to the end.
        3. If the string ends with 'aa', replace it with 'cc'.
- Output: bbbacc
- Explanation:
    - In the first iteration, rule 1 is applied to the string abbac, resulting in cabac
    - In the second interation, rule 1 doesn't apply, but rule 2 is applied to the string cabac, resulting in bbbacc
    - In the third iteration, none of the rules (1, 2, 3) apply, so the process is terminated, and the final answer is bbbacc

Transform the following string according to the above list of rules:
{string}
"""


@dataclass
class StringManipulationConfig:
    """Configuration for String Insertion dataset generation"""

    min_string_length: int = 5  # Minimum string length
    max_string_length: int = 20  # Maximum string length
    min_num_rules: int = 3  # Minimum number of rules/transforms
    max_num_rules: int = 8  # Maximum number of rules/transforms

    size: int = 500  # Virtual dataset size
    seed: Optional[int] = None

    def validate(self):
        """Validate configuration parameters"""
        assert 5 <= self.min_string_length, "Minimum string length should be at least 5"
        assert self.min_string_length <= self.max_string_length, "Minimum string length should be less than maximum"
        assert 3 <= self.min_num_rules, "Minimum number of rules should be at least 3"
        assert self.min_num_rules <= self.max_num_rules, "Minimum number of rules should be less than maximum"


class StringManipulationDataset(ProceduralDataset):
    """Generates String Insertion exercises with configurable difficulty"""

    def __init__(self, config: StringManipulationConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)
        self.vocabulary = ["a", "b", "c"]
        self.rules = [
            (
                "If the string prefix is 'ab', replace it with 'ca'.",
                lambda s: ("ca" + s[2:], 1) if s.startswith("ab") else (s, 0),
            ),
            (
                "If the string suffix is 'ac', replace it with 'cb'.",
                lambda s: (s[:-2] + "cb", 2) if s.endswith("ac") else (s, 0),
            ),
            (
                "If the string prefix is 'bc', delete the first two characters and append 'aa' to the end.",
                lambda s: (s[2:] + "aa", 3) if s.startswith("bc") else (s, 0),
            ),
            (
                "If the string suffix is 'bb', delete the last two characters.",
                lambda s: (s[:-2], 4) if s.endswith("bb") else (s, 0),
            ),
            (
                "If the string prefix is 'cb', replace it with 'aa' and delete the last character.",
                lambda s: ("aa" + s[2:-1], 5) if s.startswith("cb") and len(s) > 1 else (s, 0),
            ),
            (
                "If the string prefix is 'ca', replace it with 'bb' and append 'c' to the end.",
                lambda s: ("bb" + s[2:] + "c", 6) if s.startswith("ca") else (s, 0),
            ),
            (
                "If the string suffix is 'cc', replace it with 'b' and prepend 'a' to the start.",
                lambda s: ("a" + s[:-2] + "b", 7) if s.endswith("cc") else (s, 0),
            ),
            (
                "If the string prefix is 'aa', remove the first character.",
                lambda s: (s[1:], 8) if s.startswith("aa") else (s, 0),
            ),
            (
                "If the string contains 'abc', replace the first occurrence with 'cab'.",
                lambda s: (s.replace("abc", "cab", 1), 9) if "abc" in s else (s, 0),
            ),
            (
                "If the string contains 'bca', delete the first occurrence entirely.",
                lambda s: (s.replace("bca", "", 1), 10) if "bca" in s else (s, 0),
            ),
            (
                "If the string ends with 'ba', replace it with 'ab'.",
                lambda s: (s[:-2] + "ab", 11) if s.endswith("ba") else (s, 0),
            ),
            (
                "If the string starts with 'cc', remove the first two characters.",
                lambda s: (s[2:], 12) if s.startswith("cc") else (s, 0),
            ),
            (
                "If the string contains 'acb', replace the first occurrence with its reverse ('bca').",
                lambda s: (s.replace("acb", "bca", 1), 13) if "acb" in s else (s, 0),
            ),
            (
                "If the string ends with 'ca', remove the last character.",
                lambda s: (s[:-1], 14) if s.endswith("ca") and len(s) > 0 else (s, 0),
            ),
            (
                "If the string starts with 'bb', remove the second character.",
                lambda s: (s[0] + s[2:], 15) if s.startswith("bb") and len(s) >= 2 else (s, 0),
            ),
            (
                "If the string ends with 'aa', replace it with 'cc'.",
                lambda s: (s[:-2] + "cc", 16) if s.endswith("aa") else (s, 0),
            ),
            (
                "If the string contains 'ca' (not at the start), remove the first occurrence found after the first character.",
                lambda s: (s[:idx] + s[idx + 2 :], 17) if (idx := s.find("ca", 1)) != -1 else (s, 0),
            ),
            (
                "If the string contains an even number of 'b's (and at least one 'b'), append 'ab' at the end.",
                lambda s: (s + "ab", 18) if (s.count("b") > 0 and s.count("b") % 2 == 0) else (s, 0),
            ),
            (
                "If the string length is greater than 15, remove the middle character.",
                lambda s: (s[: len(s) // 2] + s[len(s) // 2 + 1 :], 19) if len(s) > 15 else (s, 0),
            ),
            (
                "If the string starts with 'ac', replace the first two characters with 'zz'.",
                lambda s: ("zz" + s[2:], 20) if s.startswith("ac") else (s, 0),
            ),
        ]

    def _apply_rule(self, string: str, selected_rules: list[tuple[str, callable]]) -> tuple[str, int]:
        """
        Apply the first applicable rule from the list of selected rules.
        Returns a tuple containing the modified string and the rule index (1-based) that was applied.
        If no rule is applicable, returns (s, 0).
        """
        for _, rule_fn in selected_rules:
            new_string, op_idx = rule_fn(string)
            if op_idx != 0:
                return new_string, op_idx
        return string, 0

    def _get_all_transforms(self, string: str, selected_rules: list[tuple[str, callable]]) -> list[str]:
        """
        Repeatedly apply transformation rules to a string until no further transformations can be performed,
        or a state is repeated. If a state is repeated, the process is terminated, and the state is not added to the list.
        Returns a list of string states from the initial string to the final state (i.e. the desired answer).
        """
        states = [string]
        while True:
            new_string, op_idx = self._apply_rule(states[-1], selected_rules)
            if op_idx == 0 or new_string in states:
                break
            states.append(new_string)
        return states

    def __getitem__(self, idx: int) -> dict:
        """Generate a single String Insertion question"""
        rng = Random(self.seed + idx)

        string_length = rng.randint(self.config.min_string_length, self.config.max_string_length)
        string = "".join(rng.choice(self.vocabulary) for _ in range(string_length))

        num_rules = rng.randint(self.config.min_num_rules, self.config.max_num_rules)
        selected_rules = rng.sample(self.rules, num_rules)
        rules_str = "\n".join(f"{i+1}. {rule}" for i, (rule, _) in enumerate(selected_rules))

        states = self._get_all_transforms(string, selected_rules)
        answer = states[-1]

        return {
            "question": QUESTION_TEMPLATE.format(string=string, rules=rules_str),
            "answer": str(answer),
            "metadata": {
                "string": string,
                "solution": answer,
                "states": states,
                "selected_rules": [rule for rule, _ in selected_rules],
            },
        }


register_dataset("string_manipulation", StringManipulationDataset, StringManipulationConfig)
