import os

import pandas as pd


def load_csv(file_path: str) -> pd.DataFrame:
    """
    Verifies the existence of a CSV file and loads it into a pandas DataFrame.

    :param file_path: The absolute or relative path to the CSV file.
    :return: A pandas DataFrame containing the data from the CSV file.
    :raises FileNotFoundError: If the specified file path does not exist.
    :raises IOError: If there is an error reading or parsing the CSV file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise IOError(f"Error reading CSV file: {e}")
