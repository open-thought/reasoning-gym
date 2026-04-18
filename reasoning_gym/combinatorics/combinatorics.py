import math
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

from ..coaching import BaseCurriculum, RangeAttributeDefinition
from ..factory import ProceduralDataset, register_dataset

DATASET_NAME = "combinatorics"

TASK_TYPES = (
    "ncr",
    "npr",
    "permutations_repetition",
    "inclusion_exclusion",
    "stars_and_bars",
    "pigeonhole",
    "multinomial",
    "grid_paths",
    "constrained_selection",
    "circular_permutation",
    "geometric_counting",
    "dictionary_rank",
    "derangement",
    "group_division",
    "legendres_formula",
    "integral_solutions",
)


@dataclass
class CombinatoricsConfig:
    min_n: int = 5
    max_n: int = 15
    task_types: tuple[str, ...] = TASK_TYPES
    task_weights: list[float] = field(
        default_factory=lambda: [
            0.08, 0.06, 0.08, 0.08, 0.06, 0.04,
            0.07, 0.07, 0.07, 0.07, 0.07, 0.07,
            0.06, 0.06, 0.06, 0.06,
        ]
    )
    seed: Optional[int] = None
    size: int = 500

    def validate(self) -> None:
        assert self.size > 0, "size must be positive"
        assert self.min_n >= 2, "min_n must be >= 2"
        assert self.max_n >= self.min_n, "max_n must be >= min_n"
        assert len(self.task_types) > 0, "must have at least one task type"
        assert all(t in TASK_TYPES for t in self.task_types), f"invalid task type"
        assert len(self.task_weights) == len(self.task_types), "weights must match types"


class CombinatoricsDataset(ProceduralDataset):
    def __init__(self, config: CombinatoricsConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _make_ncr(self, rng: random.Random) -> dict:
        n = rng.randint(self.config.min_n, self.config.max_n)
        k = rng.randint(1, n - 1)
        answer = math.comb(n, k)
        question = (
            f"How many ways can you choose {k} items from a set of {n} items? " f"Give your answer as a single integer."
        )
        return {"question": question, "answer": str(answer), "task_type": "ncr"}

    def _make_npr(self, rng: random.Random) -> dict:
        n = rng.randint(self.config.min_n, self.config.max_n)
        k = rng.randint(1, min(n - 1, 6))
        answer = math.perm(n, k)
        question = (
            f"How many ways can you arrange {k} items chosen from {n} distinct items "
            f"(order matters)? Give your answer as a single integer."
        )
        return {"question": question, "answer": str(answer), "task_type": "npr"}

    def _make_permutations_repetition(self, rng: random.Random) -> dict:
        letters = []
        num_distinct = rng.randint(2, 4)
        pool = "ABCDEFGH"
        chosen = rng.sample(pool, num_distinct)
        counts = {}
        for ch in chosen:
            c = rng.randint(1, 4)
            counts[ch] = c
            letters.extend([ch] * c)
        rng.shuffle(letters)
        word = "".join(letters)

        numerator = math.factorial(len(word))
        denominator = 1
        for c in counts.values():
            denominator *= math.factorial(c)
        answer = numerator // denominator

        count_desc = ", ".join(f"'{k}' appears {v} time(s)" for k, v in sorted(counts.items()))
        question = (
            f"How many distinct arrangements can be made from the letters of '{word}'? "
            f"({count_desc}) Give your answer as a single integer."
        )
        return {"question": question, "answer": str(answer), "task_type": "permutations_repetition"}

    def _make_inclusion_exclusion(self, rng: random.Random) -> dict:
        total = rng.randint(50, 200)
        a_count = rng.randint(total // 4, total * 3 // 4)
        b_count = rng.randint(total // 4, total * 3 // 4)
        max_both = min(a_count, b_count, total)
        min_both = max(0, a_count + b_count - total)
        both = rng.randint(min_both, max_both)
        neither = total - (a_count + b_count - both)

        activity_a = rng.choice(["play soccer", "like tea", "study math", "read fiction"])
        activity_b = rng.choice(["play chess", "like coffee", "study science", "read poetry"])

        question = (
            f"In a group of {total} people, {a_count} {activity_a}, {b_count} {activity_b}, "
            f"and {both} do both. How many people do neither? "
            f"Give your answer as a single integer."
        )
        return {"question": question, "answer": str(neither), "task_type": "inclusion_exclusion"}

    def _make_stars_and_bars(self, rng: random.Random) -> dict:
        n = rng.randint(self.config.min_n, self.config.max_n)
        k = rng.randint(2, 5)
        answer = math.comb(n + k - 1, k - 1)
        question = (
            f"How many ways can you distribute {n} identical balls into {k} distinct boxes "
            f"(each box can hold any number of balls)? Give your answer as a single integer."
        )
        return {"question": question, "answer": str(answer), "task_type": "stars_and_bars"}

    def _make_pigeonhole(self, rng: random.Random) -> dict:
        boxes = rng.randint(3, 20)
        extra = rng.randint(1, 10)
        items = boxes * extra + rng.randint(1, boxes - 1)
        answer = (items + boxes - 1) // boxes  # ceiling division

        question = (
            f"If {items} items are placed into {boxes} boxes, what is the minimum number of items "
            f"that must be in at least one box? Give your answer as a single integer."
        )
        return {"question": question, "answer": str(answer), "task_type": "pigeonhole"}

    # --- Advanced Counting Principles ---

    def _make_multinomial(self, rng: random.Random) -> dict:
        num_vars = rng.randint(2, 4)
        n = rng.randint(self.config.min_n, self.config.max_n)
        parts = self._random_partition(rng, n, num_vars)
        var_names = ["x", "y", "z", "w"][:num_vars]

        numerator = math.factorial(n)
        denominator = 1
        for p in parts:
            denominator *= math.factorial(p)
        answer = numerator // denominator

        term_strs = [f"{v}^{e}" for v, e in zip(var_names, parts)]
        sum_str = " + ".join(var_names)
        question = (
            f"What is the coefficient of {' * '.join(term_strs)} in the expansion of "
            f"({sum_str})^{n}? Give your answer as a single integer."
        )
        return {"question": question, "answer": str(answer), "task_type": "multinomial"}

    @staticmethod
    def _random_partition(rng: random.Random, n: int, k: int) -> list[int]:
        """Generate a random composition of n into k positive parts."""
        if k == 1:
            return [n]
        cuts = sorted(rng.sample(range(1, n), k - 1))
        parts = [cuts[0]] + [cuts[i] - cuts[i - 1] for i in range(1, len(cuts))] + [n - cuts[-1]]
        return parts

    def _make_grid_paths(self, rng: random.Random) -> dict:
        m = rng.randint(2, self.config.max_n)
        n = rng.randint(2, self.config.max_n)
        answer = math.comb(m + n, m)
        question = (
            f"How many shortest paths are there from the top-left corner to the bottom-right corner "
            f"of a {m} x {n} grid, if you can only move right or down? "
            f"Give your answer as a single integer."
        )
        return {"question": question, "answer": str(answer), "task_type": "grid_paths"}

    def _make_constrained_selection(self, rng: random.Random) -> dict:
        total_men = rng.randint(3, max(4, self.config.max_n))
        total_women = rng.randint(3, max(4, self.config.max_n))
        committee_size = rng.randint(3, min(total_men + total_women - 1, 8))
        min_women = rng.randint(1, min(total_women, committee_size - 1))

        answer = 0
        for w in range(min_women, min(total_women, committee_size) + 1):
            men_needed = committee_size - w
            if men_needed > total_men:
                continue
            answer += math.comb(total_women, w) * math.comb(total_men, men_needed)

        question = (
            f"A committee of {committee_size} people is to be formed from {total_men} men and "
            f"{total_women} women. If at least {min_women} woman/women must be included, how many "
            f"ways can the committee be formed? Give your answer as a single integer."
        )
        return {"question": question, "answer": str(answer), "task_type": "constrained_selection"}

    # --- Special Permutations & Geometry ---

    def _make_circular_permutation(self, rng: random.Random) -> dict:
        n = rng.randint(self.config.min_n, self.config.max_n)
        identical_rotations = rng.choice([True, False])

        if identical_rotations:
            answer = math.factorial(n - 1) // 2
            question = (
                f"How many distinct ways can {n} people be seated around a circular table, "
                f"where clockwise and counter-clockwise arrangements are considered the same? "
                f"Give your answer as a single integer."
            )
        else:
            answer = math.factorial(n - 1)
            question = (
                f"How many distinct ways can {n} people be seated around a circular table? "
                f"Give your answer as a single integer."
            )
        return {"question": question, "answer": str(answer), "task_type": "circular_permutation"}

    def _make_geometric_counting(self, rng: random.Random) -> dict:
        sub_type = rng.choice(["triangles", "diagonals"])
        if sub_type == "triangles":
            n = rng.randint(max(6, self.config.min_n), max(7, self.config.max_n))
            m = rng.randint(3, n - 3)
            answer = math.comb(n, 3) - math.comb(m, 3)
            question = (
                f"There are {n} points in a plane, of which {m} are collinear. "
                f"How many distinct triangles can be formed using these points as vertices? "
                f"Give your answer as a single integer."
            )
        else:
            n = rng.randint(max(4, self.config.min_n), max(5, self.config.max_n))
            answer = n * (n - 3) // 2
            question = (
                f"How many diagonals does a {n}-sided convex polygon have? "
                f"Give your answer as a single integer."
            )
        return {"question": question, "answer": str(answer), "task_type": "geometric_counting"}

    def _make_dictionary_rank(self, rng: random.Random) -> dict:
        length = rng.randint(3, min(6, max(4, self.config.max_n)))
        letters = sorted(rng.sample("ABCDEFGHIJKLMNOPQRSTUVWXYZ", length))
        word_letters = letters[:]
        rng.shuffle(word_letters)
        word = "".join(word_letters)

        rank = 1
        remaining = sorted(word_letters)
        for i, ch in enumerate(word):
            pos = remaining.index(ch)
            rank += pos * math.factorial(len(remaining) - 1)
            remaining.pop(pos)

        question = (
            f"If all permutations of the letters {', '.join(sorted(set(word)))} are arranged "
            f"in alphabetical (dictionary) order, what is the rank (position) of the word '{word}'? "
            f"Give your answer as a single integer."
        )
        return {"question": question, "answer": str(rank), "task_type": "dictionary_rank"}

    # --- Distribution & Partitioning ---

    def _make_derangement(self, rng: random.Random) -> dict:
        n = rng.randint(self.config.min_n, min(self.config.max_n, 12))
        answer = self._subfactorial(n)
        question = (
            f"How many derangements (permutations where no element appears in its original position) "
            f"are there of a set of {n} elements? Give your answer as a single integer."
        )
        return {"question": question, "answer": str(answer), "task_type": "derangement"}

    @staticmethod
    def _subfactorial(n: int) -> int:
        if n == 0:
            return 1
        if n == 1:
            return 0
        d_prev2, d_prev1 = 1, 0
        for i in range(2, n + 1):
            d_curr = (i - 1) * (d_prev1 + d_prev2)
            d_prev2, d_prev1 = d_prev1, d_curr
        return d_prev1

    def _make_group_division(self, rng: random.Random) -> dict:
        num_groups = rng.randint(2, 4)
        n = rng.randint(max(self.config.min_n, num_groups * 2), max(self.config.min_n + 1, self.config.max_n))
        group_sizes = self._random_partition(rng, n, num_groups)
        group_sizes.sort(reverse=True)

        numerator = math.factorial(n)
        denominator = 1
        for g in group_sizes:
            denominator *= math.factorial(g)
        size_counts = Counter(group_sizes)
        for cnt in size_counts.values():
            if cnt > 1:
                denominator *= math.factorial(cnt)
        answer = numerator // denominator

        sizes_str = ", ".join(str(s) for s in group_sizes)
        question = (
            f"In how many ways can {n} distinct items be divided into unlabeled groups of sizes "
            f"{sizes_str}? Give your answer as a single integer."
        )
        return {"question": question, "answer": str(answer), "task_type": "group_division"}

    # --- Number Theory in Combinatorics ---

    def _make_legendres_formula(self, rng: random.Random) -> dict:
        n = rng.randint(self.config.min_n, self.config.max_n)
        primes = [p for p in [2, 3, 5, 7, 11, 13] if p <= n]
        if not primes:
            primes = [2]
        p = rng.choice(primes)

        exponent = 0
        pk = p
        while pk <= n:
            exponent += n // pk
            pk *= p

        question = (
            f"What is the largest power of {p} that divides {n}!? "
            f"In other words, find the largest k such that {p}^k divides {n}!. "
            f"Give your answer as a single integer (the value of k)."
        )
        return {"question": question, "answer": str(exponent), "task_type": "legendres_formula"}

    def _make_integral_solutions(self, rng: random.Random) -> dict:
        r = rng.randint(2, 5)
        variant = rng.choice(["non_negative", "positive"])
        n = rng.randint(max(self.config.min_n, r), self.config.max_n)

        if variant == "non_negative":
            answer = math.comb(n + r - 1, r - 1)
            var_list = " + ".join(f"x{i+1}" for i in range(r))
            question = (
                f"How many non-negative integer solutions are there to the equation "
                f"{var_list} = {n}? Give your answer as a single integer."
            )
        else:
            answer = math.comb(n - 1, r - 1)
            var_list = " + ".join(f"x{i+1}" for i in range(r))
            question = (
                f"How many positive integer solutions are there to the equation "
                f"{var_list} = {n}? Give your answer as a single integer."
            )
        return {"question": question, "answer": str(answer), "task_type": "integral_solutions"}

    def __getitem__(self, idx: int) -> dict:
        rng = random.Random(self.seed + idx)
        task_type = rng.choices(self.config.task_types, weights=self.config.task_weights, k=1)[0]

        generators = {
            "ncr": self._make_ncr,
            "npr": self._make_npr,
            "permutations_repetition": self._make_permutations_repetition,
            "inclusion_exclusion": self._make_inclusion_exclusion,
            "stars_and_bars": self._make_stars_and_bars,
            "pigeonhole": self._make_pigeonhole,
            "multinomial": self._make_multinomial,
            "grid_paths": self._make_grid_paths,
            "constrained_selection": self._make_constrained_selection,
            "circular_permutation": self._make_circular_permutation,
            "geometric_counting": self._make_geometric_counting,
            "dictionary_rank": self._make_dictionary_rank,
            "derangement": self._make_derangement,
            "group_division": self._make_group_division,
            "legendres_formula": self._make_legendres_formula,
            "integral_solutions": self._make_integral_solutions,
        }
        result = generators[task_type](rng)
        return {
            "question": result["question"],
            "answer": result["answer"],
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "task_type": result["task_type"],
                "difficulty": {"min_n": self.config.min_n, "max_n": self.config.max_n},
            },
        }


class CombinatoricsCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(CombinatoricsCurriculum.__name__, CombinatoricsConfig)
        self._define_attributes(
            RangeAttributeDefinition(
                name="n_range",
                levels=[5, 10, 20, 30],
                lower_field_name="min_n",
                upper_field_name="max_n",
                description="Range for n in combinatorial problems",
            ),
        )


register_dataset(DATASET_NAME, CombinatoricsDataset, CombinatoricsConfig, CombinatoricsCurriculum)
