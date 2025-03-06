"""Mahjong Puzzle Generator

https://github.com/yongchao98/CodeSteer-v1.0/blob/main/create_dataset/create_dataset_mahjong.py
"""

import string
from dataclasses import dataclass
from random import Random
from typing import Optional

from ..coaching import AttributeType, BaseCurriculum, RangeAttributeDefinition
from ..factory import ProceduralDataset, register_dataset

QUESTION_TEMPLATE = """There are several letter cards, and the game rules are as follows:
1. Initially, there are 13 cards.
2. Each time, a new card is added, and a result is determined. Then, one card is removed.
3. When there are two identical cards in hand, and the newly added card is the same as these two cards, the result is determined as "Peng".
4. If there are two cards in hand such that the new card can form a consecutive letter sequence with these two cards, the result is determined as "Chi". For example: ABC, BCD, CDE, etc.
5. If the new card does not meet the conditions of 3 and 4, the result is determined as "Pass".
6. "Peng" takes precedence over "Chi".
7. The card that is removed does not affect the result determination of the current round.

Your output should be one of the following: "Peng", "Chi", or "Pass" (without quotes).

Now, given the initial cards {cards}, what is the result at the end of performing the following rounds of operations:
{operations}
"""


@dataclass
class MahjongPuzzleConfig:
    """Configuration for Mahjong Puzzle dataset generation"""

    min_num_rounds: int = 10
    max_num_rounds: int = 50

    size: int = 500  # Virtual dataset size
    seed: Optional[int] = None

    def validate(self):
        """Validate configuration parameters"""
        assert 1 <= self.min_num_rounds, "min_num_rounds must be greater than 0"
        assert self.min_num_rounds <= self.max_num_rounds, "min_num_rounds must be less than max_num_rounds"


class MahjongPuzzleDataset(ProceduralDataset):
    """Generates Mahjong Puzzle exercises with configurable difficulty"""

    def __init__(self, config: MahjongPuzzleConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)
        self.vocabulary = list(string.ascii_uppercase)
        self.k = 13

    def _get_initial_string(self, rng: Random) -> str:
        """Generate a random string of letters"""
        pool = self.vocabulary * 2  # ensure at most 2 of each letter in initial string
        characters = rng.sample(pool, self.k)
        return "".join(characters)

    def _check_peng(self, cards: str, new_card: str) -> bool:
        """Check if a Peng pattern exists with the new card"""
        return cards.count(new_card) + 1 >= 3

    def _check_chi(self, cards: str, new_card: str) -> bool:
        """Check if a Chi pattern exists with the new card"""
        all_cards = sorted(list(cards + new_card))
        for i in range(len(all_cards) - 2):
            seq = all_cards[i : i + 3]
            if ord(seq[1]) == ord(seq[0]) + 1 and ord(seq[2]) == ord(seq[1]) + 1 and new_card in seq:
                return True
        return False

    def _simulate_game(self, rng: Random, cards: str, num_rounds: int) -> tuple[str, list]:
        """
        Simulate a game of Mahjong Puzzle

        Returns:
        - result: The final result of the game
        - rounds: List of operations (add/remove) in each round
        """

        result, rounds = None, []

        for _ in range(num_rounds):
            # Try to create interesting patterns, such as Peng or Chi
            round_outcome = rng.choice(["Peng", "Chi", "Pass"])
            if round_outcome == "Peng" and any(self._check_peng(cards, c) for c in self.vocabulary):
                new_card = rng.choice([c for c in self.vocabulary if self._check_peng(cards, c)])
                result = "Peng"
            elif round_outcome == "Chi" and any(self._check_chi(cards, c) for c in self.vocabulary):
                new_card = rng.choice([c for c in self.vocabulary if self._check_chi(cards, c)])
                result = "Chi"
            else:
                new_card = rng.choice(self.vocabulary)
                result = "Pass"

            # Update states
            remove_card = rng.choice(cards)
            cards = cards.replace(remove_card, "", 1) + new_card
            rounds.append({"add": new_card, "remove": remove_card, "cards": cards, "result": result})

        return result, rounds

    def _get_article(self, card: str) -> str:
        return "an" if card[0] in "AEIOU" else "a"

    def __getitem__(self, idx: int) -> dict:
        """Generate a single Mahjong Puzzle question"""
        rng = Random(self.seed + idx)

        cards = self._get_initial_string(rng)
        num_rounds = rng.randint(self.config.min_num_rounds, self.config.max_num_rounds)
        answer, rounds = self._simulate_game(rng, cards, num_rounds)
        operations = "\n".join(
            f"Round {i+1}: Add {self._get_article(r['add'])} {r['add']} card and remove {self._get_article(r['remove'])} {r['remove']} card."
            for i, r in enumerate(rounds)
        )

        return {
            "question": QUESTION_TEMPLATE.format(cards=cards, operations=operations),
            "answer": answer,
            "metadata": {"rounds": rounds, "solution": answer},
        }


class MahjongPuzzleCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(MahjongPuzzleCurriculum.__name__, MahjongPuzzleConfig)

        # Define attributes
        self._define_attributes(
            RangeAttributeDefinition(
                name="num_rounds",
                levels=[10, 50, 100, 500],
                default_level=0,
                description="Number of rounds in the game",
                attr_type=AttributeType.APPEND,
                min_value=1,
                lower_field_name="min_num_rounds",
                upper_field_name="max_num_rounds",
            )
        )


register_dataset("mahjong_puzzle", MahjongPuzzleDataset, MahjongPuzzleConfig)
