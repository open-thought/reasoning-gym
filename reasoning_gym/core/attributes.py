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
    default_level: int
    description: str
    attr_type: AttributeType = AttributeType.STATIC  # Default to static

    @classmethod
    def validate_attributes(cls, attributes: Dict[str, 'AttributeDefinition'], valid_types: Set[AttributeType], curriculum: str) -> None:
        """
        Validates that all attributes use types from the valid_types set.

        Args:
            attributes: Dictionary of attribute definitions
            valid_types: Set of allowed AttributeTypes for this curriculum
            curriculum: A string identifier for the curriculum or class that owns these attributes

        Raises:
            ValueError: If any attribute uses an invalid type or has invalid configuration
        """
        if not valid_types:
            raise ValueError(f"Curriculum {curriculum} has no valid attribute types defined")

        if not attributes:
            raise ValueError(f"Curriculum {curriculum} has no attributes defined")

        for name, attr in attributes.items():
            # Check attribute type is valid
            if attr.attr_type not in valid_types:
                curriculum_class = f"{curriculum}." if curriculum else ""
                raise ValueError(
                    f"Attribute '{curriculum_class}{name}' uses type {attr.attr_type.value} "
                    f"which is not in the curriculum's valid types: {[t.value for t in valid_types]}"
                )

            # Check levels exist
            if not attr.levels:
                raise ValueError(f"Attribute '{curriculum}.{name}' has no levels defined")

            # Check default level is valid
            if not 0 <= attr.default_level < len(attr.levels):
                raise ValueError(
                    f"Invalid default level: {attr.default_level} for attribute '{curriculum}.{name}'. "
                    f"Must be between 0 and {len(attr.levels)-1}"
                )

    @classmethod
    def check_attribute_exists(cls, attributes: Dict[str, 'AttributeDefinition'], attr_name: str, curriculum: str) -> 'AttributeDefinition':
        """
        Check if attribute exists and return its definition.

        Args:
            attributes: Dictionary of attribute definitions
            attr_name: Name of the attribute to check
            curriculum: Name of the curriculum

        Returns:
            The AttributeDefinition for the attribute

        Raises:
            KeyError: If attribute doesn't exist
        """
        if attr_name not in attributes:
            raise KeyError(f"Attribute '{curriculum}.{attr_name}' does not exist")
        return attributes[attr_name]

    @classmethod
    def validate_level(cls, attr: 'AttributeDefinition', level: int, attr_name: str, curriculum: str) -> None:
        """
        Validate that a level is valid for an attribute.

        Args:
            attr: The attribute definition
            level: Level to validate
            attr_name: Name of the attribute
            curriculum: Name of the curriculum

        Raises:
            ValueError: If level is invalid
        """
        # TODO: if > set as [-1], if <0 set as [0]
        if not 0 <= level < len(attr.levels):
            raise ValueError(
                f"Invalid level: {level} for attribute '{curriculum}.{attr_name}'. "
                f"Must be between 0 and {len(attr.levels)-1}"
            )

    @classmethod
    def get_level_value(cls, attr: 'AttributeDefinition', level: int, attr_name: str, curriculum: str) -> Any:
        """
        Get the value for an attribute at a specific level based on its type.

        Args:
            attr: The attribute definition
            level: Level to get value for
            attr_name: Name of the attribute
            curriculum: Name of the curriculum

        Returns:
            Value for the attribute based on its level and type
        """
        if attr.attr_type == AttributeType.STATIC:
            return attr.levels[level]
        elif attr.attr_type == AttributeType.UBOUND:
            return attr.levels[level]
        elif attr.attr_type == AttributeType.APPEND:
            return attr.levels[:level + 1]

        raise ValueError(f"Unknown attribute type: {attr.attr_type} for attribute '{curriculum}.{attr_name}'")