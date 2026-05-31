"""File-loading helpers used by training and evaluation pipelines."""

import os

import pandas as pd
from pandas import DataFrame
from pandas.io.parsers import TextFileReader


def load_csv(file_path: str) -> TextFileReader | DataFrame:
    """
    Verifies the existence of a CSV file and loads it into a pandas DataFrame.

    Args:
        file_path: Absolute or relative path to the CSV file.

    Returns:
        Parsed CSV contents.

    Raises:
        FileNotFoundError: If the specified path does not exist.
        IOError: If reading or parsing the CSV fails.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise IOError(f"Error reading CSV file: {e}")
