from dataclasses import dataclass
from random import Random, seed
from typing import Dict, List, Optional, Tuple

from .contrib.logic_puzzle.generate import generate_puzzle

from ..factory import ProceduralDataset, register_dataset

@dataclass
class ZebraConfig:
    """Configuration for zebra puzzle generation"""

    k: int = 4
    m: int = 4
    seed: Optional[int] = None
    size: int = 500

    def validate(self):
        """Validate configuration parameters"""
        assert 2 <= self.k <= 7, "k must be between 2 and 7"
        assert 2 <= self.m <= 7, "m must be between 2 and 7"

class ZebraDataset(ProceduralDataset):
    """Generates Game of Life games with configurable parameters"""

    def __init__(self, config: ZebraConfig):
        self._prompt_templates = [
            "What will this Game of Life board look like after {simulation_steps} steps of simulation?\n\n{board}"
        ]

        super().__init__(config=config, seed=config.seed, size=config.size)

    def __getitem__(self, idx: int) -> dict:
        """Generate a single Zebra task

        Returns:
            dict with keys:
                - question: str, the task description
                - answer: str, a solution string
                - metadata: dict with generation parameters
        """
        seed(self.seed + idx)

        K = self.config.k
        M = self.config.m
        instance, puzzle = generate_puzzle(K, M, "train")
        q = instance['questions'][0]['question']
        a = instance['questions'][0]['answer']
        question = str(puzzle) + '\n' + q

        return {
            "question": question,
            "answer": a,
            "metadata": {
                "K": K,
                "M": M,
                "answer": a

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
