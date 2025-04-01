from dataclasses import dataclass
from typing import Any, Optional


@dataclass(kw_only=True)
class AttributeDefinition:
    name: str
    levels: list
    default_level: int = 0
    description: Optional[str] = None

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

    def get_level_value(self, level: int) -> Any:
        """
        Get the value for an attribute at a specific level based on its type.
        Args:
            attr: The attribute definition
            level: Level to get value for
        Returns:
            Value for the attribute based on its level and type
        """
        return self.levels[level]


@dataclass(kw_only=True)
class ScalarAttributeDefinition(AttributeDefinition):
    field_name: str


@dataclass(kw_only=True)
class RangeAttributeDefinition(AttributeDefinition):
    lower_field_name: str
    upper_field_name: str
    ensure_interval: bool = False  # When True, ensures the range is always an interval between two distinct values
