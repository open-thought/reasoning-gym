from typing import Any, Iterable, Optional

from ..factory import ConfigT
from .attributes import AttributeDefinition, RangeAttributeDefinition, ScalarAttributeDefinition


class BaseCurriculum:
    def __init__(self, name: str, config_cls: ConfigT):
        self.name = name
        self._config_cls = config_cls
        self._attributes: dict[str, AttributeDefinition] = {}
        self._current_levels: dict[str, int] = {}

    def generate_configuration(self, defaults: Optional[dict[str, Any]] = None) -> ConfigT:
        config_args = defaults.copy() if defaults is not None else {}
        for attr in self._attributes.values():
            if isinstance(attr, RangeAttributeDefinition):
                vals = self.get_attr_value(attr.name)
                config_args[attr.lower_field_name] = min(vals)
                config_args[attr.upper_field_name] = max(vals)
            elif isinstance(attr, ScalarAttributeDefinition):
                val = self.get_attr_value(attr.name)
                config_args[attr.field_name] = val
        print(config_args)
        return self._config_cls(**config_args)

    @property
    def attributes(self) -> dict[str, AttributeDefinition]:
        """Get the curriculum's attributes"""
        return self._attributes

    def get_attribute(self, attr_name: str) -> AttributeDefinition:
        if attr_name not in self._attributes:
            raise KeyError(f"Attribute '{self.name}.{attr_name}' does not exist")
        return self._attributes[attr_name]

    def _define_attributes(self, *attrs: tuple[AttributeDefinition, ...]) -> None:
        for attr in attrs:
            if attr.name in self.attributes:
                raise RuntimeError(f"Attribute with name {attr.name} is already defined.")
            self.attributes[attr.name] = attr

    def get_attr_level(self, attr_name: str) -> int:
        """
        Get the current level for an attribute.
        Args:
            attr_name: Name of the attribute
        Returns:
            Current level index for the attribute
        """
        attr = self.get_attribute(attr_name)
        return self._current_levels.get(attr_name, attr.default_level)

    def get_attr_value(self, attr_name: str) -> Any:
        """
        Get the current value for an attribute based on its level.
        Args:
            attr_name: Name of the attribute
        Returns:
            Current value for the attribute based on its level and type
        """
        attr = self.get_attribute(attr_name)
        level = self.get_attr_level(attr_name)
        return attr.get_level_value(level, curriculum=self.name)

    def set_attr_level(self, attr_name: str, level: int) -> None:
        """
        Set the level for an attribute.
        Args:
            attr_name: Name of the attribute
            level: New level index
        """
        attr = self.get_attribute(attr_name)
        attr.validate_level(level, curriculum=self.name)
        self._current_levels[attr_name] = level

    def increment_attr_level(self, attr_name: str) -> bool:
        """
        Increment the level of an attribute if possible.
        Args:
            attr_name: Name of the attribute to increment
        Returns:
            bool: True if level was incremented, False if already at max level
        Raises:
            KeyError: If attribute doesn't exist
        """
        attr = self.get_attribute(attr_name)
        current_level = self.get_attr_level(attr_name)
        if current_level < len(attr.levels) - 1:
            self.set_attr_level(attr_name, current_level + 1)
            return True
        return False

    def decrement_attr_level(self, attr_name: str) -> bool:
        """
        Decrement the level of an attribute if possible.
        Args:
            attr_name: Name of the attribute to decrement
        Returns:
            bool: True if level was decremented, False if already at min level
        Raises:
            KeyError: If attribute doesn't exist
        """
        current_level = self.get_attr_level(attr_name)
        if current_level > 0:
            self.set_attr_level(attr_name, current_level - 1)
            return True
        return False
