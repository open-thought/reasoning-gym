from typing import Any, Optional, TypeVar

from .attributes import AttributeDefinition, RangeAttributeDefinition, ScalarAttributeDefinition

ConfigT = TypeVar("ConfigT")


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

    def get_max_level(self) -> int:
        """
        Get the maximum level currently set across all attributes.

        Returns:
            int: The maximum level currently set across all attributes
        """
        if not self._attributes:
            return 0

        return max(self.get_attr_level(attr_name) for attr_name in self._attributes)

    def set_global_level(self, level: int) -> None:
        """
        Set all attributes to the specified level.
        If the level exceeds the number of defined levels for an attribute,
        use the highest defined level for that attribute.

        Args:
            level: The level to set for all attributes
        """
        for attr_name, attr in self._attributes.items():
            # Use the highest defined level if the requested level exceeds available levels
            attr_level = min(level, len(attr.levels) - 1)
            self.set_attr_level(attr_name, attr_level)

    def increment_global_level(self) -> bool:
        """
        Increment the level of all attributes by one from the current maximum level.

        Returns:
            bool: True if at least one attribute's level was incremented, False otherwise
        """
        current_max = self.get_max_level()
        target_level = current_max + 1

        # Check if any attribute can be incremented
        can_increment = any(
            self.get_attr_level(attr_name) < len(self._attributes[attr_name].levels) - 1
            for attr_name in self._attributes
        )

        if can_increment:
            for attr_name, attr in self._attributes.items():
                # Only increment if the attribute is not already at its maximum level
                if self.get_attr_level(attr_name) < len(attr.levels) - 1:
                    # Don't exceed the attribute's maximum level
                    new_level = min(target_level, len(attr.levels) - 1)
                    self.set_attr_level(attr_name, new_level)
            return True
        return False

    def decrement_global_level(self) -> bool:
        """
        Decrement the level of all attributes by one from the current maximum level.

        Returns:
            bool: True if at least one attribute's level was decremented, False otherwise
        """
        current_max = self.get_max_level()

        if current_max > 0:
            target_level = current_max - 1
            for attr_name in self._attributes:
                # Only decrement if the attribute is at the current maximum level
                if self.get_attr_level(attr_name) == current_max:
                    self.set_attr_level(attr_name, target_level)
            return True
        return False
