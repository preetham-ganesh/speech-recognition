import os
import json

import tensorflow as tf

from typing import Dict, Any


def check_directory_path_existence(directory_path: str) -> str:
    """Creates the directory path.

    Creates the absolute path for the directory path given in argument if it does not already exist.

    Args:
        directory_path: A string for the directory path that needs to be created if it does not already exist.

    Returns:
        A string for the absolute directory path.
    """
    # Asserts type of arguments.
    assert isinstance(
        directory_path, str
    ), "Variable directory_path should be of type 'str'."

    # Creates the following directory path if it does not exist.
    home_directory_path = os.getcwd()
    absolute_directory_path = os.path.join(home_directory_path, directory_path)
    if not os.path.isdir(absolute_directory_path):
        os.makedirs(absolute_directory_path)
    return absolute_directory_path
