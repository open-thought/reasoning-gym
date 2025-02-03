from dataclasses import dataclass
from random import Random
from typing import Dict, Optional

from ..factory import ProceduralDataset, register_dataset
from .contrib.logic_puzzle.generate import generate_puzzle


@dataclass
class ZebraConfig:
    """Configuration for zebra puzzle generation"""

    num_people: int = 4
    num_characteristics: int = 4
    seed: Optional[int] = None
    size: int = 500

    def validate(self):
        """Validate configuration parameters"""
        assert 2 <= self.num_people <= 7, "num_people must be between 2 and 7"
        assert 2 <= self.num_characteristics <= 7, "num_characteristics must be between 2 and 7"


class ZebraDataset(ProceduralDataset):
    """Generates [Zebra Puzzles](https://en.wikipedia.org/wiki/Zebra_Puzzle) with configurable parameters"""

    def __init__(self, config: ZebraConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def __getitem__(self, idx: int) -> dict:
        """Generate a single Zebra task

        Returns:
            dict with keys:
                - question: str, the task description
                - answer: str, a solution string
                - metadata: dict with generation parameters
        """
        rng = Random(self.seed + idx)

        K = self.config.num_people
        M = self.config.num_characteristics
        instance, puzzle = generate_puzzle(rng, K, M)
        q = instance["questions"][0]["question"]
        answer = instance["questions"][0]["answer"]
        question = str(puzzle) + "\n" + q

        return {
            "question": question,
            "answer": answer,
            "metadata": {
                "num_people": K,
                "num_characteristics": M,
            },
        }

    def score_answer(self, answer: Optional[str], entry: Dict[str, any]) -> float:
        """Determine if the solution provided solves the Zebra task.

        The function awards 1.0 for a correct answer.

        Args:
            answer (Optional[str]): The user's answer.
            entry (Dict[str, any]): The original dataset entry containing the correct answer.

        Returns:
            float: The computed score between 0.0 and 1.0.
        """

        if answer == None:
            return 0.0
        if answer.lower().replace("\n", "") != entry["answer"].lower().replace("\n", ""):
            return 0.01
        else:
            return 1.0  # Yay


register_dataset("zebra_puzzles", ZebraDataset, ZebraConfig)
