"""Experiment class combining dataset, scoreboard and curriculum."""

from typing import Any, Optional

from ..composite import CompositeConfig, CompositeDataset, DatasetSpec
from ..factory import create_curriculum
from ..version_manager import DatasetVersionManager
from .coach import ScoreBoard
from .curriculum_config import CurriculumExperimentConfig


class Experiment:
    def __init__(self, name: str, composite: CompositeDataset):
        self.name = name
        self.composite = composite
        self.score_board = ScoreBoard()

    def get_dataset_entry(self, index: int) -> dict:
        return self.composite[index]

    def score_answer_with_id(
        self, answer: Optional[str], entry_id: str, conversation: Optional[list[dict]] = None
    ) -> float:
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
    def __init__(self, name: str, config: CurriculumExperimentConfig, size: int, seed: Optional[int] = None):
        """Initialize curriculum experiment with configured datasets and their curricula.

        Args:
            name: Name of the experiment
            config: Configuration specifying datasets and their attribute levels
            size: Number of examples to generate
            seed: Random seed for reproducibility
        """
        # Initialize curricula and build dataset specs
        self.curricula = {}
        dataset_specs = []

        # Process each dataset in the curriculum config
        for dataset_name, attr_config in config.curricula.items():
            # Create and store curriculum
            curriculum = create_curriculum(dataset_name)
            self.curricula[dataset_name] = curriculum

            # Handle special "*" attribute that sets all levels
            if "*" in attr_config.attribute_levels:
                level = attr_config.attribute_levels["*"]
                for attr_name in curriculum.attributes:
                    curriculum.set_attr_level(attr_name, level)

            # Set individual attribute levels (overriding "*" if specified)
            for attr_name, level in attr_config.attribute_levels.items():
                if attr_name != "*":
                    curriculum.set_attr_level(attr_name, level)

            # Generate dataset config from curriculum
            dataset_config = curriculum.generate_configuration()

            # Create dataset spec
            spec = DatasetSpec(name=dataset_name, weight=attr_config.weight, config=dataset_config.__dict__)
            dataset_specs.append(spec)

        # Create composite config with all datasets
        composite_config = CompositeConfig(size=size, seed=seed, datasets=dataset_specs)

        # Create composite dataset
        version_manager = DatasetVersionManager()
        composite = CompositeDataset(config=composite_config, version_manager=version_manager)

        # Initialize base experiment
        super().__init__(name=name, composite=composite)

        # Store curriculum config
        self.curriculum_config = config

    def update_difficulty(self):
        """Update difficulty levels based on performance metrics"""
        # TODO: Implement difficulty adjustment logic
        pass
