from dataclasses import dataclass
from typing import Dict, Optional

import yaml


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

    @classmethod
    def from_yaml_stream(cls, stream) -> "CurriculumExperimentConfig":
        """Load configuration from a YAML stream

        Args:
            stream: A file-like object containing YAML data

        Returns:
            CurriculumExperimentConfig instance

        Raises:
            ValueError: If YAML data has invalid format
        """
        data = yaml.safe_load(stream)

        if not isinstance(data, dict):
            raise ValueError("YAML data must contain a dictionary")

        if "curricula" not in data:
            raise ValueError("YAML data must contain a 'curricula' key")

        # Convert curriculum configs
        curricula = {}
        for dataset_name, config in data["curricula"].items():
            if not isinstance(config, dict):
                raise ValueError(f"Curriculum config for {dataset_name} must be a dictionary")

            if "attribute_levels" not in config:
                raise ValueError(f"Curriculum config for {dataset_name} must contain 'attribute_levels'")

            weight = config.get("weight", 1.0)
            curricula[dataset_name] = CurriculumAttributeConfig(
                attribute_levels=config["attribute_levels"], weight=weight
            )

        return cls(curricula=curricula)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "CurriculumExperimentConfig":
        """Load configuration from YAML file

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            CurriculumExperimentConfig instance

        Raises:
            ValueError: If YAML file has invalid format
        """
        with open(yaml_path, "r") as f:
            return cls.from_yaml_stream(f)
