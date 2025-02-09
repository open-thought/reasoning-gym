"""Package containing data files used by reasoning_gym"""

from importlib import resources
from pathlib import Path
from typing import Any, Dict
import json


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
    return resources.files("reasoning_gym.data").joinpath(filename)


def read_data_file(filename: str) -> str:
    """Read the contents of a data file in the package.

    Args:
        filename: Name of the file in the data directory

    Returns:
        String contents of the file

    Example:
        >>> content = read_data_file("pg19362.txt")
    """
    return resources.files("reasoning_gym.data").joinpath(filename).read_text()

def read_json_file(file_name: str) -> Dict[str, Any]:
    """Read a json file from a file path into a dictionary."""
    file_path = get_data_file_path(file_name)
    with open(file_path, "r") as file:
        return json.load(file)


__all__ = ["get_data_file_path", "read_data_file"]
