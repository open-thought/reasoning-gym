from collections import abc
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Optional


class AttributeType(StrEnum):
    """Defines how attribute levels should be interpreted"""

    STATIC = "static"  # Each level is independent
    UBOUND = "ubound"  # Each level is an upper bound
    APPEND = "append"  # Each level includes all previous levels


@dataclass(kw_only=True)
class AttributeDefinition:
    name: str
    levels: list
    default_level: int
    description: Optional[str] = None
    attr_type: AttributeType = AttributeType.STATIC  # Default to static
    min_value: Optional[int | float] = None  # Minimum value for numeric attributes

    def validate_level(self, level: int, curriculum: str) -> None:
        """
        Validate that a level is valid for an attribute.
        Args:
            level: Level to validate
            curriculum: Name of the curriculum
        Raises:
            ValueError: If level is invalid
        """
        # TODO: if > set as [-1], if <0 set as [0]
        if not 0 <= level < len(self.levels):
            raise ValueError(
                f"Invalid level: {level} for attribute '{curriculum}.{self.name}'. "
                f"Must be between 0 and {len(self.levels)-1}"
            )

    def get_level_value(self, level: int, curriculum: str) -> Any:
        """
        Get the value for an attribute at a specific level based on its type.
        Args:
            attr: The attribute definition
            level: Level to get value for
        Returns:
            Value for the attribute based on its level and type
        """
        if self.attr_type == AttributeType.STATIC:
            return self.levels[level]
        elif self.attr_type == AttributeType.UBOUND:
            return self.levels[level]
        elif self.attr_type == AttributeType.APPEND:
            return self.levels[: level + 1]

        raise ValueError(f"Unknown attribute type: {self.attr_type} for attribute '{curriculum}.{self.name}'")


@dataclass(kw_only=True)
class ScalarAttributeDefinition(AttributeDefinition):
    field_name: str


@dataclass(kw_only=True)
class RangeAttributeDefinition(AttributeDefinition):
    lower_field_name: str
    upper_field_name: str

    def get_level_value(self, level: int, curriculum: str) -> Any:
        v = super().get_level_value(level, curriculum)
        if not isinstance(v, abc.Iterable):
            return [v]
        return v
