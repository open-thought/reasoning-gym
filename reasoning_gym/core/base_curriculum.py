"""
Base class for exercise curricula that defines the interface and common functionality.
"""

from typing import Dict, List, Any
from dataclasses import dataclass
from reasoning_gym.core.attributes import AttributeDefinition, AttributeType

@dataclass
class Template:
    """Defines a template for generating questions and answers"""
    question: str
    answer: str
    metadata: Dict[str, Any]

class BaseCurriculum:
    """Base class for all exercise curricula"""

    def __init__(self, name: str):
        self.name = name
        self._attributes: Dict[str, AttributeDefinition] = {}
        self._templates: List[Template] = []
        self._valid_types: set[AttributeType] = set()
        self._current_levels: Dict[str, int] = {}
        
        # Let child class fill in the structure
        self._init_curriculum()
        
        # Validate the filled structure
        self._validate()

    # TODO: Why?
    def _init_curriculum(self) -> None:
        """
        Initialize curriculum-specific attributes and templates.
        Must be implemented by subclasses to fill in the pre-defined structure.
        """
        raise NotImplementedError("Subclasses must implement _init_curriculum()")

    def _validate(self) -> None:
        """Validate the curriculum configuration"""
        # Validate attributes
        AttributeDefinition.validate_attributes(
            self._attributes,
            self._valid_types,
            curriculum=self.name
        )
        # Validate templates exist
        if not self._templates:
            raise ValueError(f"Curriculum {self.name} has no templates defined")

    @property
    def attributes(self) -> Dict[str, AttributeDefinition]:
        """Get the curriculum's attributes"""
        return self._attributes

    @property
    def templates(self) -> List[Template]:
        """Get the curriculum's templates"""
        return self._templates
    
    def get_attr_level(self, attr_name: str) -> int:
        """
        Get the current level for an attribute.
        
        Args:
            attr_name: Name of the attribute
            
        Returns:
            Current level index for the attribute
        """
        attr = AttributeDefinition.check_attribute_exists(self._attributes, attr_name, self.name)
        return self._current_levels.get(attr_name, attr.default_level)
    
    def get_attr_value(self, attr_name: str) -> Any:
        """
        Get the current value for an attribute based on its level.
        
        Args:
            attr_name: Name of the attribute
            
        Returns:
            Current value for the attribute based on its level and type
        """
        attr = AttributeDefinition.check_attribute_exists(self._attributes, attr_name, self.name)
        level = self.get_attr_level(attr_name)
        return AttributeDefinition.get_level_value(attr, level, attr_name, self.name)
    
    def set_attr_level(self, attr_name: str, level: int) -> None:
        """
        Set the level for an attribute.
        
        Args:
            attr_name: Name of the attribute
            level: New level index
        """
        attr = AttributeDefinition.check_attribute_exists(self._attributes, attr_name, self.name)
        AttributeDefinition.validate_level(attr, level, attr_name, self.name)
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
        attr = AttributeDefinition.check_attribute_exists(self._attributes, attr_name, self.name)
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
        attr = AttributeDefinition.check_attribute_exists(self._attributes, attr_name, self.name)
        current_level = self.get_attr_level(attr_name)
        
        if current_level > 0:
            self.set_attr_level(attr_name, current_level - 1)
            return True
        return False
