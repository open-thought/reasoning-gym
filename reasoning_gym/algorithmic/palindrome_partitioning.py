"""Given a string, return all possible partitions of the string such that each substring is a palindrome.

A popular Leetcode problem:
https://leetcode.com/problems/palindrome-partitioning/description/
"""

import json
import re
from dataclasses import dataclass
from random import Random
from typing import Dict, Optional

from ..data import read_data_file
from ..factory import ProceduralDataset, register_dataset

QUESTION_TEMPLATE = """Given a string, partition it such that every substring is a palindrome.

A palindrome is a word that reads the same backward as forward.

You may return all possible palindrome partitioning in any order.

Example:
Input: "aab"
Output: [["a","a","b"],["aa","b"]]

Partition the following string into palindromes:
{string}
"""


@dataclass
class PalindromePartitioningConfig:
    """Configuration for Palindrome Partitioning dataset generation"""

    size: int = 500  # Virtual dataset size
    seed: Optional[int] = None

    def validate(self):
        """Validate configuration parameters"""
        pass


class PalindromePartitioningDataset(ProceduralDataset):
    """Generates Palindrome Partitioning exercises with configurable difficulty"""

    def __init__(self, config: PalindromePartitioningConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)
        self.words = [
            re.sub(r"\W+", "", word.strip()) for word in read_data_file("in_the_year_2889.txt").split() if word.strip()
        ]

    def __len__(self) -> int:
        return self.config.size

    def __iter__(self):
        self._current_idx = 0
        return self

    def __next__(self):
        if self._current_idx >= self.config.size:
            raise StopIteration
        item = self[self._current_idx]
        self._current_idx += 1
        return item

    def _sort_list(self, lst: list[list[str]]) -> list[list[str]]:
        """Sort the list of palindrome partitions"""
        return sorted([sublist for sublist in lst], key=lambda x: x[0] if x else "")

    def _palindrome_partitioning(self, string: str) -> list[list[str]]:
        """Return all possible palindrome partitions of a string"""
        if not string:
            return []
        dp = {}

        def is_palindrome(i, j) -> bool:
            if i >= j:
                return True
            if (i, j) in dp:
                return dp[(i, j)]
            dp[(i, j)] = string[i] == string[j] and is_palindrome(i + 1, j - 1)
            return dp[(i, j)]

        res, temp = [], []

        def _partition(idx) -> None:
            if idx >= len(string):
                res.append(temp[:])
            for i in range(idx, len(string)):
                if is_palindrome(idx, i):
                    temp.append(string[idx : i + 1])
                    _partition(i + 1)
                    temp.pop()

        _partition(0)
        return self._sort_list(res)

    def score_answer(self, answer: Optional[str], entry: Dict[str, any]) -> float:
        """Score a single Palindrome Partitioning question"""
        reward = 0
        if answer is not None:
            try:
                answer = json.loads(answer)
                oracle = entry["metadata"]["solution"]
                answer_str = json.dumps(self._sort_list(answer))
                oracle_str = json.dumps(self._sort_list(oracle))
                if answer_str == oracle_str:
                    reward = 1
                else:
                    reward = 0.01
            except Exception:
                reward = 0
        return reward

    def __getitem__(self, idx: int) -> dict:
        """Generate a single Palindrome Partitioning question"""
        rng = Random(self.seed + idx)
        string = rng.choice(self.words)
        answer = self._palindrome_partitioning(string)
        answer_str = json.dumps(answer)

        return {
            "question": QUESTION_TEMPLATE.format(string=string),
            "answer": answer_str,
            "metadata": {"string": string, "solution": answer},
        }


register_dataset("palindrome_partitioning", PalindromePartitioningDataset, PalindromePartitioningConfig)
