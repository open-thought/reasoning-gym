from dataclasses import dataclass
from random import Random
from typing import Any, Dict, List, Optional

import yaml

from .dataset import ProceduralDataset
from .factory import create_dataset, register_dataset


@dataclass
class DatasetSpec:
    """Specification for a single dataset within the composite"""

    name: str
    weight: float
    config: dict

    def validate(self):
        """Validate dataset specification"""
        assert self.name, "Dataset name cannot be empty"
        assert self.weight > 0, "Weight must be positive"
        assert isinstance(self.config, dict), "Config must be a dictionary"


@dataclass
class CompositeConfig:
    """Configuration for CompositeDataset"""

    size: int = 500
    seed: Optional[int] = None
    datasets: List[DatasetSpec] = None

    def validate(self):
        """Validate configuration parameters"""
        assert self.size > 0, "size must be positive"
        assert self.datasets, "Must specify at least one dataset"
        assert len(self.datasets) > 0, "Must specify at least one dataset"

        # Validate each dataset spec
        for ds in self.datasets:
            ds.validate()

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "CompositeConfig":
        """Load configuration from YAML file"""
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        # Convert dataset specs to DatasetSpec objects
        if "datasets" in data:
            data["datasets"] = [DatasetSpec(**ds) for ds in data["datasets"]]

        return cls(**data)


class CompositeDataset(ProceduralDataset):
    """A dataset that combines multiple datasets with weighted sampling"""

    def __init__(self, config: CompositeConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

        # Initialize sub-datasets with incremented seeds
        self.datasets = {}
        self.weights = []
        total_weight = 0.0

        for i, ds_spec in enumerate(config.datasets):
            # Create dataset with derived seed
            ds_config = ds_spec.config.copy()
            if "seed" not in ds_config:
                ds_config["seed"] = self.seed + i + 1
            if "size" not in ds_config:
                ds_config["size"] = self.size

            self.datasets[ds_spec.name] = create_dataset(ds_spec.name, **ds_config)
            total_weight += ds_spec.weight
            self.weights.append(ds_spec.weight)

        # Normalize weights
        self.weights = [w / total_weight for w in self.weights]
        self.dataset_names = [ds.name for ds in config.datasets]

    def __getitem__(self, idx: int) -> dict:
        """Generate a single dataset item by sampling from sub-datasets"""
        # Create deterministic RNG for this index
        rng = Random(self.seed + idx)

        # Sample dataset according to weights
        dataset_idx = rng.choices(range(len(self.dataset_names)), weights=self.weights, k=1)[0]
        dataset_name = self.dataset_names[dataset_idx]
        dataset = self.datasets[dataset_name]

        # Get item from selected dataset
        item = dataset[idx]

        # Add source dataset info to metadata
        item["metadata"]["source_dataset"] = dataset_name
        item["metadata"]["source_index"] = idx

        return item

    def update_dataset_config(self, dataset_name: str, config_updates: Dict[str, Any]) -> None:
        """Update configuration of a specific dataset

        Args:
            dataset_name: Name of the dataset to update
            config_updates: Dictionary of configuration parameters to update

        Raises:
            KeyError: If dataset_name is not found
            AttributeError: If config parameter doesn't exist
        """
        if dataset_name not in self.datasets:
            raise KeyError(f"Dataset '{dataset_name}' not found")

        dataset = self.datasets[dataset_name]

        # Create new config with updates
        new_config = dataset.config.__class__(**vars(dataset.config))
        for key, value in config_updates.items():
            setattr(new_config, key, value)

        # Validate new config
        new_config.validate()

        # Create new dataset instance with updated config
        dataset_cls = dataset.__class__
        self.datasets[dataset_name] = dataset_cls(new_config)

    def score_answer(self, answer: Optional[str], entry: Dict[str, Any]) -> float:
        """Forward scoring to appropriate dataset"""
        dataset_name = entry["metadata"]["source_dataset"]
        return self.datasets[dataset_name].score_answer(answer, entry)


# Register the dataset
register_dataset("composite", CompositeDataset, CompositeConfig)
