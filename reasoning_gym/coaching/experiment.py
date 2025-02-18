"""Experiment class combining dataset, scoreboard and curriculum."""

from dataclasses import dataclass
from typing import Optional

from ..composite import CompositeConfig, CompositeDataset
from .coach import ScoreBoard


@dataclass
class Experiment:
    """
    An experiment combines a dataset with scoring and curriculum management.
    
    Attributes:
        name: Unique identifier for the experiment
        dataset: The composite dataset for generating examples
        score_board: Tracks performance metrics
        config: The configuration used to create the dataset
    """
    
    name: str
    dataset: CompositeDataset
    score_board: ScoreBoard
    config: CompositeConfig

    @classmethod
    def create(cls, name: str, config: CompositeConfig) -> "Experiment":
        """Create a new experiment from a configuration."""
        dataset = CompositeDataset(config)
        score_board = ScoreBoard()
        return cls(
            name=name,
            dataset=dataset,
            score_board=score_board,
            config=config
        )
