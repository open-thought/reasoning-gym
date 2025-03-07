"""Group all anagrams together in a list.

Anagram is a word formed by rearranging the letters of a different word, using all the original letters exactly once.

A popular Leetcode problem:
https://leetcode.com/problems/group-anagrams/description/
"""

import json
from collections import defaultdict
from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from ..coaching import AttributeType, BaseCurriculum, RangeAttributeDefinition
from ..data import get_data_file_path
from ..factory import ProceduralDataset, register_dataset

QUESTION_TEMPLATE = """An anagram is a word formed by rearranging the letters of a different word, using all the original letters exactly once.

Your job is to group the anagrams together. You can return the answer in any order.

The output is a list of lists of strings, where each outer list contains a group of anagrams, e.g. [["eat", "tea"], ["tan", "nat"]].

Group the following list of words into anagrams:
{words}
"""


@dataclass
class GroupAnagramsConfig:
    """Configuration for Group Anagrams dataset generation"""

    min_anagram_groups: int = 2  # Minimum number of anagram groups present in the input
    max_anagram_groups: int = 10  # Maximum number of anagram groups present in the input
    min_words_per_group: int = 2  # Minimum number of words in a single anagram group
    max_words_per_group: int = 5  # Maximum number of words in a single anagram group

    size: int = 500  # Virtual dataset size
    seed: Optional[int] = None

    def validate(self):
        """Validate configuration parameters"""
        assert 2 <= self.min_anagram_groups <= self.max_anagram_groups, "Invalid number of anagram groups"
        assert 2 <= self.min_words_per_group <= self.max_words_per_group, "Invalid number of words per group"


class GroupAnagramsDataset(ProceduralDataset):
    """Generates Group Anagrams exercises with configurable difficulty"""

    def __init__(self, config: GroupAnagramsConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)
        with get_data_file_path("anagrams.jsonl").open() as f:
            self.anagrams = [json.loads(line)["words"] for line in f]

    def _get_anagram_words(self, rng: Random, num_groups: int) -> list[str]:
        """Generate a list of words with anagrams"""
        words = []
        for sample in rng.sample(self.anagrams, num_groups):
            num_words = min(len(sample), rng.randint(self.config.min_words_per_group, self.config.max_words_per_group))
            anagrams = rng.sample(sample, num_words)
            words.extend(anagrams)
        return words

    def _sort_nested_list(self, lst: list[list[str]]) -> list[list[str]]:
        """Sort a nested list of strings"""
        return sorted([sorted(sublist) for sublist in lst], key=lambda x: x[0] if x else "")

    def _group_anagrams(self, words: list[str]) -> list[list[str]]:
        """Group anagrams together"""

        def _codify(word):
            code = [0] * 26
            for c in word:
                code[ord(c) - ord("a")] += 1
            return tuple(code)

        res = defaultdict(list)
        for word in words:
            code = _codify(word)
            res[code].append(word)

        anagrams = list(res.values())
        return self._sort_nested_list(anagrams)

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """Score a single Group Anagrams question"""
        reward = 0.0
        if answer is not None:
            try:
                answer = json.loads(answer)
                oracle = entry["metadata"]["solution"]
                answer_str = json.dumps(self._sort_nested_list(answer))
                oracle_str = json.dumps(self._sort_nested_list(oracle))
                if answer_str == oracle_str:
                    reward = 1.0
                else:
                    reward = 0.01  # json parsable
            except Exception:
                reward = 0.0
        return reward

    def __getitem__(self, idx: int) -> dict:
        """Generate a single Group Anagrams question"""
        rng = Random(self.seed + idx)

        anagram_groups = min(
            len(self.anagrams), rng.randint(self.config.min_anagram_groups, self.config.max_anagram_groups)
        )
        words = self._get_anagram_words(rng, num_groups=anagram_groups)
        answer = self._group_anagrams(words)
        answer_str = json.dumps(answer)

        return {
            "question": QUESTION_TEMPLATE.format(words=json.dumps(words)),
            "answer": answer_str,
            "metadata": {
                "words": words,
                "solution": answer,
                "difficulty": {
                    "anagram_groups": anagram_groups,
                },
            },
        }


class GroupAnagramsCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(GroupAnagramsCurriculum.__name__, GroupAnagramsConfig)

        # Define attributes
        self._define_attributes(
            RangeAttributeDefinition(
                name="anagram_groups",
                levels=[10, 100, 1_000, 10_000],
                default_level=0,
                description="Number of anagram groups in the input",
                attr_type=AttributeType.APPEND,
                min_value=2,
                lower_field_name="min_anagram_groups",
                upper_field_name="max_anagram_groups",
            ),
            RangeAttributeDefinition(
                name="words_per_group",
                levels=[2, 5, 10, 20],
                default_level=0,
                description="Number of words in a single anagram group",
                attr_type=AttributeType.APPEND,
                min_value=2,
                lower_field_name="min_words_per_group",
                upper_field_name="max_words_per_group",
            ),
        )


register_dataset("group_anagrams", GroupAnagramsDataset, GroupAnagramsConfig, GroupAnagramsCurriculum)
