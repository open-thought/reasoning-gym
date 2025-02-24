from dataclasses import is_dataclass
from typing import Optional, Type, TypeVar

from reasoning_gym.coaching.base_curriculum import BaseCurriculum, ConfigT

from .dataset import ProceduralDataset

# Type variables for generic type hints

DatasetT = TypeVar("DatasetT", bound=ProceduralDataset)
CurriculumT = TypeVar("CurriculumT", bound=BaseCurriculum)

# Global registry of datasets
DATASETS: dict[str, tuple[Type[ProceduralDataset], Type]] = {}
CURRICULA: dict[str, BaseCurriculum] = {}


def register_dataset(
    name: str,
    dataset_cls: Type[DatasetT],
    config_cls: Type[ConfigT],
    curriculum_cls: Optional[CurriculumT] = None,
) -> None:
    """
    Register a dataset class with its configuration class and optional curriculum.

    Args:
        name: Unique identifier for the dataset
        dataset_cls: Class derived from ProceduralDataset
        config_cls: Configuration dataclass for the dataset
        curriculum_cls: Optional curriculum class for progressive difficulty

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

    if curriculum_cls:
        CURRICULA[name] = curriculum_cls


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


def create_curriculum(name: str) -> BaseCurriculum:
    """
    Create a curriculum instance for the named dataset.

    Args:
        name: Registered dataset name

    Returns:
        Configured curriculum instance

    Raises:
        ValueError: If dataset not found or has no curriculum registered
    """
    if name not in CURRICULA:
        raise ValueError(f"No curriculum registered for dataset '{name}'")

    curriculum_cls = CURRICULA[name]

    return curriculum_cls()


def has_curriculum(name: str) -> bool:
    return name in CURRICULA
