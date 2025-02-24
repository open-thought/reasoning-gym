from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class CurriculumAttributeConfig:
    """Configuration for curriculum attribute levels"""
    # Either specify individual attribute levels
    attribute_levels: Optional[Dict[str, int]] = None
    # Or use "*" to set all attributes to same level
    all_attributes_level: Optional[int] = None
    # Weight for sampling this dataset
    weight: float = 1.0

    def validate(self):
        """Validate the configuration"""
        if self.attribute_levels is not None and self.all_attributes_level is not None:
            raise ValueError("Cannot specify both attribute_levels and all_attributes_level")
        if self.attribute_levels is None and self.all_attributes_level is None:
            raise ValueError("Must specify either attribute_levels or all_attributes_level")

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
