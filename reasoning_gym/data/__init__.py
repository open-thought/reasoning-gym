"""Package containing data files used by reasoning_gym"""

from importlib import resources
from pathlib import Path
from typing import Union

def get_data_file_path(filename: str) -> Path:
    """Get the path to a data file in the package.
    
    Args:
        filename: Name of the file in the data directory
        
    Returns:
        Path object pointing to the data file
        
    Example:
        >>> path = get_data_file_path("pg19362.txt")
        >>> with open(path) as f:
        ...     content = f.read()
    """
    with resources.path('reasoning_gym.data', filename) as data_path:
        return data_path

def read_data_file(filename: str) -> str:
    """Read the contents of a data file in the package.
    
    Args:
        filename: Name of the file in the data directory
        
    Returns:
        String contents of the file
        
    Example:
        >>> content = read_data_file("pg19362.txt")
    """
    with resources.open_text('reasoning_gym.data', filename) as f:
        return f.read()

__all__ = ['get_data_file_path', 'read_data_file']
