"""
Curriculum definition for the ChainSum exercise.
This file defines the templates, attributes, and difficulty levels for generating chain sum problems.
"""

from typing import Dict, List, Union, Any
from reasoning_gym.core.attributes import AttributeDefinition, AttributeType

# Define which attribute types are valid for this curriculum
ATTRIBUTE_TYPES = {
    AttributeType.STATIC,  # For base numbers
    AttributeType.UBOUND,  # For ranges like digits and terms
    AttributeType.APPEND   # For operators and notations
}

# Curriculum definition
CURRICULUM_NAME = "ChainSumCurriculum"

ATTRIBUTES = {
    "num_digits": AttributeDefinition(
        levels=[1, 2, 3, 4],
        current_level=0,  # Start with 1-digit numbers
        description="Number of digits in each operand",
        attr_type=AttributeType.UBOUND
    ),

    "num_decimals": AttributeDefinition(
        levels=[0, 1, 2],
        current_level=0,  # Start with integers
        description="Number of decimal places in operands",
        attr_type=AttributeType.UBOUND
    ),

    "operators": AttributeDefinition(
        levels=['+', '-', '*', '/', '**'],
        current_level=0,  # Start with basic operators
        description="Set of operators that can be used, each level includes all previous operators",
        attr_type=AttributeType.APPEND
    ),

    "max_terms": AttributeDefinition(
        levels=[2, 3, 4, 5],
        current_level=0,  # Start with 2 terms
        description="Maximum number of terms in the expression",
        attr_type=AttributeType.UBOUND
    ),

    "sign": AttributeDefinition(
        levels=['', '+', '-'],
        current_level=0,  # Start without negatives
        description="Whether negative numbers are allowed",
        attr_type=AttributeType.APPEND
    ),

	"notation": AttributeDefinition(
        levels=["regular", "scientific"],
        current_level=0,
        description="The notation to use for the expression",
        attr_type=AttributeType.APPEND
    ),

	"base": AttributeDefinition(
        levels=[10, 2, 16],
        current_level=0,
        description="The base to use for the expression",
        attr_type=AttributeType.STATIC
    )
}

# Validate attributes use allowed types (and include the curriculum name in error messages)
AttributeDefinition.validate_attributes(ATTRIBUTES, ATTRIBUTE_TYPES, curriculum=CURRICULUM_NAME)

# Template definitions
TEMPLATES = [
    {
        "question": "What is {expression}?",
        "answer": "{result}",
        "metadata": {"type": "direct"}
    },
    {
        "question": "Calculate the following: {expression}",
        "answer": "{result}",
        "metadata": {"type": "direct"}
    },
    {
        "question": "Solve {expression}",
        "answer": "{result}",
        "metadata": {"type": "direct"}
    }
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