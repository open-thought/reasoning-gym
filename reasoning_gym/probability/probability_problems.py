import math
import random
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Optional

from ..coaching import BaseCurriculum, RangeAttributeDefinition
from ..factory import ProceduralDataset, register_dataset

DATASET_NAME = "probability_problems"

TASK_TYPES = (
    "independent_events",
    "compound_events",
    "total_probability",
    "bayes_theorem",
    "binomial_probability",
    "binomial_stats",
    "geometric_series",
    "geometric_region",
    "expectation_variance",
)


@dataclass
class ProbabilityProblemsConfig:
    min_n: int = 3
    max_n: int = 10
    task_types: tuple[str, ...] = TASK_TYPES
    task_weights: list[float] = field(
        default_factory=lambda: [
            0.12, 0.11, 0.12, 0.12,
            0.11, 0.10, 0.11, 0.10, 0.11,
        ]
    )
    seed: Optional[int] = None
    size: int = 500

    def validate(self) -> None:
        assert self.size > 0, "size must be positive"
        assert self.min_n >= 2, "min_n must be >= 2"
        assert self.max_n >= self.min_n, "max_n must be >= min_n"
        assert len(self.task_types) > 0, "must have at least one task type"
        assert all(t in TASK_TYPES for t in self.task_types), "invalid task type"
        assert len(self.task_weights) == len(self.task_types), "weights must match types"


class ProbabilityProblemsDataset(ProceduralDataset):
    def __init__(self, config: ProbabilityProblemsConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _rand_prob(self, rng: random.Random) -> Fraction:
        """Generate a random probability as a simple fraction in (0, 1)."""
        denom = rng.choice([2, 3, 4, 5, 6, 8, 10])
        numer = rng.randint(1, denom - 1)
        return Fraction(numer, denom)

    # --- Section 1: Conditional Probability & Multiplication Theorem ---

    def _make_independent_events(self, rng: random.Random) -> dict:
        pa = self._rand_prob(rng)
        pb = self._rand_prob(rng)
        variant = rng.choice(["intersection", "union", "neither"])

        if variant == "intersection":
            answer = pa * pb
            question = (
                f"Events A and B are independent with P(A) = {pa} and P(B) = {pb}. "
                f"What is P(A and B)? Give your answer as a simplified fraction."
            )
        elif variant == "union":
            answer = pa + pb - pa * pb
            question = (
                f"Events A and B are independent with P(A) = {pa} and P(B) = {pb}. "
                f"What is P(A or B)? Give your answer as a simplified fraction."
            )
        else:
            answer = (1 - pa) * (1 - pb)
            question = (
                f"Events A and B are independent with P(A) = {pa} and P(B) = {pb}. "
                f"What is the probability that neither A nor B occurs? "
                f"Give your answer as a simplified fraction."
            )
        return {"question": question, "answer": str(answer), "task_type": "independent_events"}

    def _make_compound_events(self, rng: random.Random) -> dict:
        total = rng.randint(max(4, self.config.min_n), max(4, self.config.max_n))
        color_a_count = rng.randint(2, total - 2)
        color_b_count = total - color_a_count

        colors = ["red", "blue", "green", "white", "black"]
        color_a = rng.choice(colors)
        color_b = rng.choice([c for c in colors if c != color_a])

        seq = rng.choice(["ab", "ba"])
        if seq == "ab":
            prob = Fraction(color_a_count, total) * Fraction(color_b_count, total - 1)
            seq_desc = f"the first is {color_a} and the second is {color_b}"
        else:
            prob = Fraction(color_b_count, total) * Fraction(color_a_count, total - 1)
            seq_desc = f"the first is {color_b} and the second is {color_a}"

        question = (
            f"A bag contains {color_a_count} {color_a} balls and {color_b_count} {color_b} balls. "
            f"You draw 2 balls one after another without replacement. "
            f"What is the probability that {seq_desc}? "
            f"Give your answer as a simplified fraction."
        )
        return {"question": question, "answer": str(prob), "task_type": "compound_events"}

    # --- Section 2: Total Probability & Bayes' Theorem ---

    def _make_total_probability(self, rng: random.Random) -> dict:
        num_bags = rng.randint(2, 3)
        bags = []
        for _ in range(num_bags):
            red = rng.randint(1, self.config.max_n)
            blue = rng.randint(1, self.config.max_n)
            bags.append((red, blue))

        p_bag = Fraction(1, num_bags)
        p_red = Fraction(0)
        for red, blue in bags:
            p_red += p_bag * Fraction(red, red + blue)

        bag_desc = ". ".join(
            f"Bag {i + 1} contains {r} red and {b} blue balls"
            for i, (r, b) in enumerate(bags)
        )
        question = (
            f"{bag_desc}. "
            f"One bag is chosen uniformly at random and a ball is drawn from it. "
            f"What is the probability the ball is red? "
            f"Give your answer as a simplified fraction."
        )
        return {"question": question, "answer": str(p_red), "task_type": "total_probability"}

    def _make_bayes_theorem(self, rng: random.Random) -> dict:
        num_bags = rng.randint(2, 3)
        bags = []
        for _ in range(num_bags):
            red = rng.randint(1, self.config.max_n)
            blue = rng.randint(1, self.config.max_n)
            bags.append((red, blue))

        target_bag = rng.randint(0, num_bags - 1)

        p_bag = Fraction(1, num_bags)
        p_red = Fraction(0)
        for red, blue in bags:
            p_red += p_bag * Fraction(red, red + blue)

        red_t, blue_t = bags[target_bag]
        p_red_given_target = Fraction(red_t, red_t + blue_t)
        p_target_given_red = (p_bag * p_red_given_target) / p_red

        bag_desc = ". ".join(
            f"Bag {i + 1} contains {r} red and {b} blue balls"
            for i, (r, b) in enumerate(bags)
        )
        question = (
            f"{bag_desc}. "
            f"One bag is chosen uniformly at random and a ball is drawn. The ball is red. "
            f"What is the probability that it came from Bag {target_bag + 1}? "
            f"Give your answer as a simplified fraction."
        )
        return {"question": question, "answer": str(p_target_given_red), "task_type": "bayes_theorem"}

    # --- Section 3: Probability Distributions ---

    def _make_binomial_probability(self, rng: random.Random) -> dict:
        p_choices = [
            Fraction(1, 6), Fraction(1, 4), Fraction(1, 3),
            Fraction(1, 2), Fraction(2, 3), Fraction(3, 4),
        ]
        p = rng.choice(p_choices)
        q = 1 - p
        n = rng.randint(self.config.min_n, min(self.config.max_n, 8))
        r = rng.randint(0, n)

        prob = Fraction(math.comb(n, r)) * (p ** r) * (q ** (n - r))
        question = (
            f"A biased coin has a probability of heads equal to {p}. "
            f"If it is flipped {n} times, what is the probability of getting exactly {r} heads? "
            f"Give your answer as a simplified fraction."
        )
        return {"question": question, "answer": str(prob), "task_type": "binomial_probability"}

    def _make_binomial_stats(self, rng: random.Random) -> dict:
        p_choices = [
            Fraction(1, 6), Fraction(1, 4), Fraction(1, 3),
            Fraction(1, 2), Fraction(2, 3), Fraction(3, 4),
        ]
        p = rng.choice(p_choices)
        q = 1 - p
        n = rng.randint(self.config.min_n, self.config.max_n)
        variant = rng.choice(["mean", "variance"])

        if variant == "mean":
            answer = Fraction(n) * p
            question = (
                f"A random variable X follows a binomial distribution with {n} trials "
                f"and success probability {p}. What is E(X), the expected value? "
                f"Give your answer as a simplified fraction or integer."
            )
        else:
            answer = Fraction(n) * p * q
            question = (
                f"A random variable X follows a binomial distribution with {n} trials "
                f"and success probability {p}. What is Var(X), the variance? "
                f"Give your answer as a simplified fraction or integer."
            )
        return {"question": question, "answer": str(answer), "task_type": "binomial_stats"}

    # --- Section 4: Geometric Probability ---

    def _make_geometric_series(self, rng: random.Random) -> dict:
        p_choices = [
            Fraction(1, 6), Fraction(1, 5), Fraction(1, 4),
            Fraction(1, 3), Fraction(1, 2),
        ]
        p = rng.choice(p_choices)
        q = rng.choice(p_choices)

        # A and B alternate; A goes first.
        # P(A wins) = p / (1 - (1-p)(1-q))  via infinite geometric series
        answer = p / (1 - (1 - p) * (1 - q))

        name_a = rng.choice(["Alice", "Arun", "Alex"])
        name_b = rng.choice(["Bob", "Bala", "Beth"])

        question = (
            f"{name_a} and {name_b} play a game where they take alternate turns, "
            f"with {name_a} going first. On each of her turns, {name_a} has a probability "
            f"of {p} of winning the game. If {name_a} does not win, {name_b} then has a "
            f"probability of {q} of winning on his turn. If neither wins, the process "
            f"repeats. What is the probability that {name_a} wins the game? "
            f"Give your answer as a simplified fraction."
        )
        return {"question": question, "answer": str(answer), "task_type": "geometric_series"}

    def _make_geometric_region(self, rng: random.Random) -> dict:
        a = rng.randint(max(2, self.config.min_n), self.config.max_n)
        variant = rng.choice(["leq", "geq"])

        if variant == "leq":
            c = rng.randint(1, a)
            answer = Fraction(c * c, 2 * a * a)
            question = (
                f"Two numbers x and y are each chosen uniformly at random from [0, {a}]. "
                f"What is the probability that x + y <= {c}? "
                f"Give your answer as a simplified fraction."
            )
        else:
            c = rng.randint(a + 1, 2 * a - 1)
            side = 2 * a - c
            answer = Fraction(side * side, 2 * a * a)
            question = (
                f"Two numbers x and y are each chosen uniformly at random from [0, {a}]. "
                f"What is the probability that x + y >= {c}? "
                f"Give your answer as a simplified fraction."
            )
        return {"question": question, "answer": str(answer), "task_type": "geometric_region"}

    # --- Section 5: Random Variables & Expectation ---

    def _make_expectation_variance(self, rng: random.Random) -> dict:
        k = rng.randint(3, 5)
        outcomes = sorted(rng.sample(range(1, 11), k))
        weights = [rng.randint(1, 10) for _ in range(k)]
        total_weight = sum(weights)
        probs = [Fraction(w, total_weight) for w in weights]

        variant = rng.choice(["expectation", "variance"])

        ex = sum(Fraction(x) * p for x, p in zip(outcomes, probs))

        if variant == "expectation":
            answer = ex
            stat_name = "E(X), the expected value"
        else:
            ex2 = sum(Fraction(x * x) * p for x, p in zip(outcomes, probs))
            answer = ex2 - ex * ex
            stat_name = "Var(X), the variance"

        table_lines = " | ".join(f"P(X={x}) = {p}" for x, p in zip(outcomes, probs))
        question = (
            f"A discrete random variable X has the following probability distribution: "
            f"{table_lines}. "
            f"What is {stat_name}? "
            f"Give your answer as a simplified fraction or integer."
        )
        return {"question": question, "answer": str(answer), "task_type": "expectation_variance"}

    def __getitem__(self, idx: int) -> dict:
        rng = random.Random(self.seed + idx)
        task_type = rng.choices(self.config.task_types, weights=self.config.task_weights, k=1)[0]

        generators = {
            "independent_events": self._make_independent_events,
            "compound_events": self._make_compound_events,
            "total_probability": self._make_total_probability,
            "bayes_theorem": self._make_bayes_theorem,
            "binomial_probability": self._make_binomial_probability,
            "binomial_stats": self._make_binomial_stats,
            "geometric_series": self._make_geometric_series,
            "geometric_region": self._make_geometric_region,
            "expectation_variance": self._make_expectation_variance,
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

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        if answer is None:
            return 0.0
        oracle = entry["answer"]
        if answer.strip() == oracle.strip():
            return 1.0
        try:
            ans_frac = Fraction(answer.strip())
            oracle_frac = Fraction(oracle.strip())
            if ans_frac == oracle_frac:
                return 1.0
            diff = abs(float(ans_frac) - float(oracle_frac))
            if diff < 1e-4:
                return 0.9
            if diff < 1e-2:
                return 0.5
            return 0.0
        except (ValueError, ZeroDivisionError):
            return 0.0


class ProbabilityProblemsCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(ProbabilityProblemsCurriculum.__name__, ProbabilityProblemsConfig)
        self._define_attributes(
            RangeAttributeDefinition(
                name="n_range",
                levels=[3, 5, 10, 15, 20],
                lower_field_name="min_n",
                upper_field_name="max_n",
                description="Range for n in probability problems",
            ),
        )


register_dataset(
    DATASET_NAME, ProbabilityProblemsDataset, ProbabilityProblemsConfig, ProbabilityProblemsCurriculum
)
