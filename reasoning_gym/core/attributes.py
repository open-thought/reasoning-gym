"""
Core definitions for curriculum attributes and types.
"""

from typing import Dict, List, Union, Any, Set, Optional
from dataclasses import dataclass
from enum import Enum

class AttributeType(Enum):
    """Defines how attribute levels should be interpreted"""
    STATIC = "static"    # Each level is independent
    UBOUND = "ubound"    # Each level is an upper bound
    APPEND = "append"    # Each level includes all previous levels

@dataclass
class AttributeDefinition:
    """Defines a difficulty attribute with its possible levels and properties"""
    levels: List[Any]
    current_level: int
    description: str
    attr_type: AttributeType = AttributeType.STATIC  # Default to static

    @classmethod
    def validate_attributes(cls, attributes: Dict[str, 'AttributeDefinition'], valid_types: Set[AttributeType], curriculum: Optional[str] = None) -> None:
        """
        Validates that all attributes use types from the valid_types set.

        Args:
            attributes: Dictionary of attribute definitions
            valid_types: Set of allowed AttributeTypes for this curriculum
            curriculum: A string identifier for the curriculum or class that owns these attributes

        Raises:
            ValueError: If any attribute uses an invalid type
        """
        for name, attr in attributes.items():
            if attr.attr_type not in valid_types:
                curriculum_class = f"{curriculum}." if curriculum else ""
                raise ValueError(
                    f"Attribute '{curriculum_class}{name}' uses type {attr.attr_type.value} "
                    f"which is not in the curriculum's valid types: {[t.value for t in valid_types]}"
                )