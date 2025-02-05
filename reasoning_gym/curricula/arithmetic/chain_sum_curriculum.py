"""
Curriculum definition for the ChainSum exercise.
"""

from typing import Dict, List, Any
from reasoning_gym.core.base_curriculum import BaseCurriculum, Template
from reasoning_gym.core.attributes import AttributeDefinition, AttributeType

class ChainSumCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__("ChainSumCurriculum")

    def _init_curriculum(self) -> None:
        """Initialize the ChainSum curriculum configuration"""
        
        # Define valid attribute types
        self._valid_types = {
            AttributeType.STATIC,  # For base numbers
            AttributeType.UBOUND,  # For ranges like digits and terms
            AttributeType.APPEND   # For operators and notations
        }
        
        # Define attributes
        self._attributes = {
            "num_digits": AttributeDefinition(
                levels=[2, 4, 10],
                default_level=0,  # Start with 1-digit numbers
                description="Number of digits in each operand",
                attr_type=AttributeType.UBOUND
                # distribution_type
            ),
            "num_decimals": AttributeDefinition(
                levels=[0, 1, 2],
                default_level=0,  # Start with integers
                description="Number of decimal places in operands",
                attr_type=AttributeType.UBOUND
            ),
            "operators": AttributeDefinition(
                levels=['+', '-', '*', '/', '**'],
                default_level=0,  # Start with basic operators
                description="Set of operators that can be used",
                attr_type=AttributeType.APPEND
            ),
            "max_terms": AttributeDefinition(
                levels=[2, 3, 4, 5],
                default_level=0,  # Start with 2 terms
                description="Maximum number of terms in the expression",
                attr_type=AttributeType.UBOUND
            ),
            "sign": AttributeDefinition(
                levels=['', '+', '-'],
                default_level=0,  # Start without negatives
                description="Whether negative numbers are allowed",
                attr_type=AttributeType.APPEND
            ),
            "notation": AttributeDefinition(
                levels=["regular", "scientific"],
                default_level=0,
                description="The notation to use for the expression",
                attr_type=AttributeType.APPEND
            ),
            "base": AttributeDefinition(
                levels=[10, 2, 16],
                default_level=0,
                description="The base to use for the expression",
                attr_type=AttributeType.STATIC
            )
        }
        
        # Define templates
        self._templates = [
            Template(
                question="What is {expression}?",
                answer="{result}",
                metadata={"type": "direct"}
            ),
            Template(
                question="Calculate the following: {expression}",
                answer="{result}",
                metadata={"type": "direct"}
            ),
            Template(
                question="Solve {expression}",
                answer="{result}",
                metadata={"type": "direct"}
            )
        ]

# Generator functions for placeholders
def generate_expression(attributes: Dict[str, Any]) -> Dict[str, str]:
    """
    Generates an expression and its result based on current attribute levels.
    This is a placeholder - actual implementation will be in the Exercise class.
    
    Args:
        attributes: Dictionary of current attribute levels
        
    Returns:
        Dict containing the expression and result as strings
    """
    # This will be implemented in the Exercise class
    pass