"""Experiment class combining dataset, scoreboard and curriculum."""

from typing import Any, Optional
from ..composite import CompositeConfig, CompositeDataset
from ..version_manager import DatasetVersionManager
from .coach import ScoreBoard


class Experiment:
    def __init__(self, name: str, composite: CompositeDataset):
        self.name = name
        self.composite = composite
        self.score_board = ScoreBoard()

    def get_dataset_entry(self, index: int) -> dict:
        return self.composite[index]

    def score_answer_with_id(self, answer: Optional[str], entry_id: str, conversation: Optional[list[dict]] = None) -> float:
        dataset, index, dataset_name = self.composite.resolve_entry_id(entry_id)
        entry = dataset[index]
        score = dataset.score_answer(answer, entry)
        metadata = entry["metadata"]
        self.score_board.add_score(score, metadata, conversation)
        return score

    @classmethod
    def create(cls, name: str, config: CompositeConfig) -> "Experiment":
        """Create a new experiment from a configuration."""
        version_manager = DatasetVersionManager()
        dataset = CompositeDataset(config, version_manager=version_manager)
        return cls(name=name, dataset=dataset)



class CurriculumExperiment(Experiment):
    def __init__(self, name: str, size: int, seed: Optional[int]):
        config = CompositeConfig(size=size, seed=seed)
        composite = CompositeDataset(config)

        super().__init__(name=name)

    def update_difficulty(self):
        pass
    