from dataclasses import dataclass
from typing import Dict


@dataclass
class CurriculumAttributeConfig:
    """Configuration for curriculum attribute levels"""

    # Dictionary mapping attribute names to levels
    # Special key "*" means apply that level to all attributes
    attribute_levels: Dict[str, int]
    # Weight for sampling this dataset
    weight: float = 1.0

    def validate(self):
        """Validate the configuration"""
        if not self.attribute_levels:
            raise ValueError("Must specify at least one attribute level")


@dataclass
class CurriculumExperimentConfig:
    """Configuration for curriculum experiments"""

    # Dictionary mapping dataset names to their curriculum configurations
    curricula: Dict[str, CurriculumAttributeConfig]

    def validate(self):
        """Validate the configuration"""
        if not self.curricula:
            raise ValueError("Must specify at least one curriculum")

        for dataset_name, attr_config in self.curricula.items():
            if not isinstance(attr_config, CurriculumAttributeConfig):
                raise ValueError(f"Invalid attribute config for dataset {dataset_name}")
            attr_config.validate()
