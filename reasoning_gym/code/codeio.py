from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from ..factory import ProceduralDataset, register_dataset


@dataclass
class CodeIOConfig:
    """Configuration for BF task generation"""

    seed: Optional[int] = None
    size: int = 500

    def validate(self) -> None:
        """Validate configuration parameters"""
        pass


class CodeIODataset(ProceduralDataset):
    def __init__(self, config: CodeIOConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

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

    def __getitem__(self, idx: int) -> dict:
        """Generate a single mini sudoku puzzle"""
        rng = Random(self.seed + idx)

        # TODO
        question = ""
        solution = ""

        return {
            "question": question,
            "answer": solution,
            "metadata": {},
        }

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        oracle_answer = entry["answer"].strip()
        reward = 0.0
        if answer is not None and len(answer) > 0:
            answer = answer.strip()
            if answer == oracle_answer:
                reward = 1.0
            elif oracle_answer in answer:
                reward = len(oracle_answer) / len(answer)
            else:
                reward = 0.01

        return reward


# Register the dataset
register_dataset("codeio", CodeIODataset, CodeIOConfig)
