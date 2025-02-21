from dataclasses import dataclass
from functools import reduce
from math import gcd
from random import Random
from typing import Any, Optional

from ..factory import ProceduralDataset, register_dataset


def compute_probability(dice, target):
    """
    Computes the probability of rolling a total of at least `target`
    when rolling dice specified in the list `dice`. Each element in dice
    is the number of sides on that die. The computation is done via dynamic programming.
    Returns the probability as a fraction (numerator, denominator) and as a float.
    """
    # dp[i][s] = number of ways to get sum s using the first i dice.
    # We use only one dictionary for the current dp state.
    dp = {0: 1}
    for sides in dice:
        new_dp = {}
        for current_sum, count in dp.items():
            # Each die gives a number from 1 to sides.
            for face in range(1, sides + 1):
                new_sum = current_sum + face
                new_dp[new_sum] = new_dp.get(new_sum, 0) + count
        dp = new_dp

    total_outcomes = reduce(lambda a, b: a * b, dice, 1)
    ways = sum(count for s, count in dp.items() if s >= target)

    # Simplify the fraction (ways / total_outcomes)
    def simplify(n, d):
        common = gcd(n, d)
        return n // common, d // common

    frac = simplify(ways, total_outcomes)
    return frac, ways / total_outcomes


def generate_puzzle(num_dice, max_dice_size, rng):
    """
    Generates a puzzle:
      - It forces one die to have max_dice_size.
      - The other (num_dice-1) dice are chosen randomly between 2 and max_dice_size-1.
      - The dice are then shuffled.
      - The target total is chosen roughly in the middle (but you can adjust the method).

    It then computes the probability of rolling a total at least the target.
    Finally, it prints out the puzzle statement and the answer.
    """

    # Guarantee one die is the maximum.
    dice = [max_dice_size]
    for _ in range(num_dice - 1):
        # Choose a die size randomly from 2 up to max_dice_size-1.
        # (If max_dice_size == 2 then all dice are 2-sided.)
        if max_dice_size > 2:
            die = rng.randint(2, max_dice_size - 1)
        else:
            die = 2
        dice.append(die)

    # Optionally, sort dice in descending order (as is common in puzzles)
    dice.sort(reverse=True)

    # Compute minimum and maximum possible totals.
    min_total = num_dice  # each die gives at least 1
    max_total = sum(dice)

    # Choose a target total. For an interesting puzzle,
    # we choose a target somewhere in the middle third of the range.
    low_target = min_total + (max_total - min_total) // 3
    high_target = min_total + 2 * (max_total - min_total) // 3
    target = rng.randint(low_target, high_target)

    # Compute probability.
    (num, den), prob = compute_probability(dice, target)

    # Create a string representing the dice, e.g., "1d20, 1d17, 1d6" etc.
    dice_str = ", ".join(f"1d{s}" for s in dice)

    # Return the puzzle.
    return {"dice_str": dice_str, "target": target, "num": num, "den": den}


@dataclass
class DiceConfig:
    """Configuration for dice puzzle generation"""

    num_dice: int = 4
    max_dice_size: int = 20
    seed: Optional[int] = None
    size: int = 500

    def validate(self):
        """Validate configuration parameters"""
        assert self.num_dice >= 1, "num_dice must be gte 1"
        assert self.max_dice_size >= 2, "max_dice_size must be gte 2"


class DiceDataset(ProceduralDataset):
    """Generates Dice-based puzzles with configurable parameters"""

    def __init__(self, config: DiceConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def __getitem__(self, idx: int) -> dict:
        """Generate a single Dice task

        Returns:
            dict with keys:
                - question: str, the task description
                - answer: str, a solution string
                - metadata: dict with generation parameters
        """
        rng = Random(self.seed + idx)
        puzzle = generate_puzzle(self.config.num_dice, self.config.max_dice_size, rng)
        puzzle_str = f"I have these dice: {puzzle['dice_str']}. What are the odds of rolling {puzzle['target']} or higher? (Assume that all dice are rolled at once, and that '1d6' represents one roll of a 6-sided dice.) Please respond with a reduced fraction representing the probability [ex., 1/60]."
        answer_str = f"{puzzle['num']}/{puzzle['den']}"

        return {
            "question": puzzle_str,
            "answer": answer_str,
            "metadata": {},
        }

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """Determine if the solution provided solves the Dice task.

        The function awards 1.0 for a correct answer.

        Args:
            answer (Optional[str]): The user's answer.
            entry (dict[str, Any]): The original dataset entry containing the correct answer.

        Returns:
            float: The computed score between 0.0 and 1.0.
        """

        if answer == None:
            return 0.0
        if answer.lower().replace("\n", "") != entry["answer"].lower().replace("\n", ""):
            return 0.01
        else:
            return 1.0  # Yay


register_dataset("dice", DiceDataset, DiceConfig)
