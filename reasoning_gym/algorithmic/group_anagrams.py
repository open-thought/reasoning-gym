"""Group all anagrams together in a list.

Anagram is a word formed by rearranging the letters of a different word, using all the original letters exactly once.

A popular Leetcode problem:
https://leetcode.com/problems/group-anagrams/description/
"""

import json
from collections import defaultdict
from dataclasses import dataclass
from random import Random
from typing import Dict, Optional

from ..data import get_data_file_path
from ..factory import ProceduralDataset, register_dataset

MAX_ANAGRAM_GROUPS = 500

QUESTION_TEMPLATE = """An anagram is a word formed by rearranging the letters of a different word, using all the original letters exactly once.

Your job is to group the anagrams together. You can return the answer in any order.

Example:
Input: ["eat", "tea", "tan", "ate", "nat", "bat"]
Output: [["bat"], ["nat", "tan"], ["ate", "eat", "tea"]]
Explanation:
    - There is no string in the input that can be rearranged to form "bat".
    - The strings "nat" and "tan" are anagrams as they can be rearranged to form each other.

Group the following list of words into anagrams:
{words}
"""


@dataclass
class GroupAnagramsConfig:
    """Configuration for Group Anagrams dataset generation"""

    anagram_groups: int = 10  # Groups of anagrams present in the input
    max_words_per_group: int = 5  # Maximum number of words in a single anagram group

    size: int = 500  # Virtual dataset size
    seed: Optional[int] = None

    def validate(self):
        """Validate configuration parameters"""
        assert (
            1 <= self.anagram_groups <= MAX_ANAGRAM_GROUPS
        ), f"anagram_groups must be between 1 and {MAX_ANAGRAM_GROUPS}"
        assert 1 <= self.max_words_per_group, "max_words_per_group must be at least 1"


class GroupAnagramsDataset(ProceduralDataset):
    """Generates Group Anagrams exercises with configurable difficulty"""

    def __init__(self, config: GroupAnagramsConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)
        with get_data_file_path("anagrams.jsonl").open() as f:
            self.anagrams = [json.loads(line)["words"] for line in f]

    def _get_anagram_words(self, rng: Random) -> list[str]:
        """Generate a list of words with anagrams"""
        words = []
        for sample in rng.sample(self.anagrams, self.config.anagram_groups):
            anagrams = rng.sample(sample, rng.randint(1, min(len(sample), self.config.max_words_per_group)))
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

    def score_answer(self, answer: Optional[str], entry: Dict[str, any]) -> float:
        """Score a single Group Anagrams question"""
        reward = 0
        if answer is not None:
            try:
                answer = json.loads(answer)
                oracle = entry["metadata"]["solution"]
                answer_str = json.dumps(self._sort_nested_list(answer))
                oracle_str = json.dumps(self._sort_nested_list(oracle))
                if answer_str == oracle_str:
                    reward = 1
                else:
                    reward = 0.01
            except Exception:
                reward = 0
        return reward

    def __getitem__(self, idx: int) -> dict:
        """Generate a single Group Anagrams question"""
        rng = Random(self.seed + idx)
        words = self._get_anagram_words(rng)
        answer = self._group_anagrams(words)
        answer_str = json.dumps(answer)

        return {
            "question": QUESTION_TEMPLATE.format(words=json.dumps(words)),
            "answer": answer_str,
            "metadata": {"words": words, "solution": answer},
        }


register_dataset("group_anagrams", GroupAnagramsDataset, GroupAnagramsConfig)
