from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from ..coaching import BaseCurriculum, ScalarAttributeDefinition
from ..factory import ProceduralDataset, register_dataset
from .contrib.logic_puzzle.generate import generate_puzzle

DATASET_NAME = "zebra_puzzles"


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
        question = question + "? Provide only the name of the person as your final answer."

        return {
            "question": question,
            "answer": answer,
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "difficulty": {"num_people": K, "num_characteristics": M},
            },
        }

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """Determine if the solution provided solves the Zebra task.

        The function awards 1.0 for a correct answer.

        Args:
            answer (Optional[str]): The user's answer.
            entry (dict[str, Any]): The original dataset entry containing the correct answer.

        Returns:
            float: The computed score between 0.0 and 1.0.
        """

        if isinstance(answer, str):
            if answer.lower().replace("\n", "") == entry["answer"].lower().replace("\n", ""):
                return 1.0  # Yay
        return 0.0


class ZebraCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(ZebraCurriculum.__name__, ZebraConfig)
        self._define_attributes(
            ScalarAttributeDefinition(
                name="num_people",
                levels=list(range(2, 8)),
                description="The number of people in the Zebra puzzle",
                field_name="num_people",
            ),
            ScalarAttributeDefinition(
                name="num_characteristics",
                levels=list(range(2, 8)),
                description="The number of characteristics in the Zebra puzzle",
                field_name="num_characteristics",
            ),
        )


register_dataset(DATASET_NAME, ZebraDataset, ZebraConfig, ZebraCurriculum)
