"""Trace a fully associative, initially empty least recently used cache."""

import json
from collections import OrderedDict
from dataclasses import dataclass
from random import Random
from typing import Optional

from ..coaching import BaseCurriculum, RangeAttributeDefinition
from ..factory import ProceduralDataset, register_dataset

DATASET_NAME = "lru_cache"

QUESTION_TEMPLATE = """Simulate an initially empty cache that holds at most {capacity} items.
The cache uses Least Recently Used (LRU) eviction:
- If an accessed item is already cached, record H (hit).
- Otherwise, record M (miss) and insert the item. If the cache is full, first evict
  the item whose most recent access was earliest.
- Every access, including a hit, makes that item the most recently used item.

Access sequence (process left to right): {accesses}

Return only a JSON object with exactly two keys:
- "trace": a string of H/M characters, one per access, with no spaces.
- "cache": the final cached item names as a list, ordered from least recently
  used to most recently used.
For example, the answer format is {{"trace": "MMH", "cache": ["P1", "P0"]}}.
"""


@dataclass
class LRUCacheConfig:
    """Bounds for capacity, access count, and the pool of available item names.

    The pool size is not a guarantee that every available item is accessed.
    hit_probability controls the chance of choosing a resident item when both
    resident and nonresident choices exist. Otherwise the available kind is used.
    """

    min_capacity: int = 2
    max_capacity: int = 5
    min_num_accesses: int = 12
    max_num_accesses: int = 30
    min_num_items: int = 6
    max_num_items: int = 12
    hit_probability: float = 0.45
    size: int = 500
    seed: Optional[int] = None

    def validate(self):
        for name in ("capacity", "num_accesses", "num_items"):
            lower = getattr(self, f"min_{name}")
            upper = getattr(self, f"max_{name}")
            assert type(lower) is int and type(upper) is int, f"{name} bounds must be integers"
            assert 1 <= lower <= upper, f"Require 1 <= min_{name} <= max_{name}"
        assert 0 <= self.hit_probability <= 1, "hit_probability must be between 0 and 1"
        assert type(self.size) is int and self.size > 0, "size must be a positive integer"


class LRUCacheDataset(ProceduralDataset):
    """Generate LRU state-tracking problems with exact, structured rewards."""

    def __init__(self, config: LRUCacheConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    @staticmethod
    def _simulate(accesses: list[str], capacity: int) -> dict:
        """Return the hit/miss trace and cache ordered least to most recent."""
        cache = OrderedDict()
        trace = []
        for item in accesses:
            if item in cache:
                trace.append("H")
                cache.move_to_end(item)
            else:
                trace.append("M")
                if len(cache) == capacity:
                    cache.popitem(last=False)
                cache[item] = None
        return {"trace": "".join(trace), "cache": list(cache)}

    def _generate_accesses(self, rng: Random, capacity: int, num_items: int, length: int) -> list[str]:
        """Mix resident and nonresident accesses without rejection sampling."""
        items = [f"P{i}" for i in range(num_items)]
        cache = []  # least to most recently used
        accesses = []
        for _ in range(length):
            nonresident = [item for item in items if item not in cache]
            if cache and (not nonresident or rng.random() < self.config.hit_probability):
                item = rng.choice(cache)
                cache.remove(item)
            else:
                item = rng.choice(nonresident)
                if len(cache) == capacity:
                    cache.pop(0)
            cache.append(item)
            accesses.append(item)
        return accesses

    def __getitem__(self, idx: int) -> dict:
        rng = Random(self.seed + idx)
        capacity = rng.randint(self.config.min_capacity, self.config.max_capacity)
        length = rng.randint(self.config.min_num_accesses, self.config.max_num_accesses)
        num_items = rng.randint(self.config.min_num_items, self.config.max_num_items)
        accesses = self._generate_accesses(rng, capacity, num_items, length)
        solution = self._simulate(accesses, capacity)
        return {
            "question": QUESTION_TEMPLATE.format(capacity=capacity, accesses=" ".join(accesses)),
            "answer": json.dumps(solution),
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "capacity": capacity,
                "num_items": num_items,
                "accesses": accesses,
                "solution": solution,
                "difficulty": {
                    "capacity": (self.config.min_capacity, self.config.max_capacity),
                    "num_accesses": (self.config.min_num_accesses, self.config.max_num_accesses),
                    "num_items": (self.config.min_num_items, self.config.max_num_items),
                    "hit_probability": self.config.hit_probability,
                },
            },
        }

    def score_answer(self, answer: Optional[str], entry: dict) -> float:
        """Accept correct JSON regardless of whitespace or key order; no partial credit."""
        if not isinstance(answer, str):
            return 0.0

        def unique_keys(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Duplicate JSON key")
                result[key] = value
            return result

        try:
            parsed = json.loads(answer, object_pairs_hook=unique_keys)
        except (ValueError, RecursionError):
            return 0.0
        if not isinstance(parsed, dict) or set(parsed) != {"trace", "cache"}:
            return 0.0
        if not isinstance(parsed["trace"], str) or not isinstance(parsed["cache"], list):
            return 0.0
        if not all(isinstance(item, str) for item in parsed["cache"]):
            return 0.0
        return float(parsed == json.loads(entry["answer"]))


class LRUCacheCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(LRUCacheCurriculum.__name__, LRUCacheConfig)
        for name, levels, description in (
            ("capacity", [2, 4, 8, 16], "Maximum number of cached items"),
            ("num_accesses", [12, 24, 48, 96], "Number of accesses to process"),
            ("num_items", [6, 12, 24, 48], "Number of available item names"),
        ):
            self._define_attributes(
                RangeAttributeDefinition(
                    name=name,
                    levels=levels,
                    description=description,
                    lower_field_name=f"min_{name}",
                    upper_field_name=f"max_{name}",
                )
            )


register_dataset(DATASET_NAME, LRUCacheDataset, LRUCacheConfig, LRUCacheCurriculum)
