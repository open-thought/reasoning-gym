"""Experiment class combining dataset, scoreboard and curriculum."""

from typing import Any, Literal, Optional

from reasoning_gym.coaching.base_curriculum import CurriculumContext

from ..composite import CompositeConfig, CompositeDataset, DatasetSpec
from ..factory import create_curriculum
from ..version_manager import DatasetVersionManager
from .curriculum_config import CurriculumExperimentConfig
from .score_board import ScoreBoard


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
        score_board_metadata = {"difficulty": metadata["difficulty"], "source_dataset": metadata["source_dataset"]}
        self.score_board.add_score(dataset_name, score, score_board_metadata, conversation)
        return score

    @classmethod
    def create(cls, name: str, config: CompositeConfig) -> "Experiment":
        """Create a new experiment from a configuration."""
        version_manager = DatasetVersionManager()
        dataset = CompositeDataset(config, version_manager=version_manager)
        return cls(name=name, dataset=dataset)


class CurriculumExperiment(Experiment):
    def __init__(
        self,
        name: str,
        config: CurriculumExperimentConfig,
        size: int,
        context: Optional[CurriculumContext] = None,
        seed: Optional[int] = None,
    ):
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
            dataset_config = curriculum.generate_configuration(context=context)

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
        self.context = context

    def update_difficulty(self, dataset_name: str, method: Literal["increment", "decrement"]):
        """Update difficulty levels based on performance metrics"""
        if method not in ["increment", "decrement"]:
            raise ValueError(f"Invalid method: {method}")

        if method == "increment":
            self.curricula[dataset_name].increment_global_level()
        elif method == "decrement":
            self.curricula[dataset_name].decrement_global_level()

        config = self.curricula[dataset_name].get_global_level()
        self.composite.update_dataset_config(dataset_name, config)
