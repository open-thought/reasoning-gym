from dataclasses import is_dataclass
from typing import Type, TypeVar

from .dataset import ProceduralDataset

# Type variables for generic type hints
ConfigT = TypeVar("ConfigT")
DatasetT = TypeVar("DatasetT", bound=ProceduralDataset)

# Global registry of datasets
DATASETS: dict[str, tuple[Type[ProceduralDataset], Type]] = {}


def register_dataset(name: str, dataset_cls: Type[DatasetT], config_cls: Type[ConfigT]) -> None:
    """
    Register a dataset class with its configuration class.

    Args:
        name: Unique identifier for the dataset
        dataset_cls: Class derived from ProceduralDataset
        config_cls: Configuration dataclass for the dataset

    Raises:
        ValueError: If name is already registered or invalid types provided
    """
    if name in DATASETS:
        raise ValueError(f"Dataset '{name}' is already registered")

    if not issubclass(dataset_cls, ProceduralDataset):
        raise ValueError(f"Dataset class must inherit from ProceduralDataset, got {dataset_cls}")

    if not is_dataclass(config_cls):
        raise ValueError(f"Config class must be a dataclass, got {config_cls}")

    DATASETS[name] = (dataset_cls, config_cls)


def create_dataset(name: str, **kwargs) -> ProceduralDataset:
    """
    Create a dataset instance by name with the given configuration.

    Args:
        name: Registered dataset name

    Returns:
        Configured dataset instance

    Raises:
        ValueError: If dataset not found or config type mismatch
    """
    if name not in DATASETS:
        raise ValueError(f"Dataset '{name}' not registered")

    dataset_cls, config_cls = DATASETS[name]

    config = config_cls(**kwargs)

    return dataset_cls(config=config)
