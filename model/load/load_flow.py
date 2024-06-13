"""Loads the parquet file and returns it as a dataframe"""

import os

import pandas as pd
from prefect import flow, task


def load() -> pd.DataFrame:
    """
    Loads a Parquet file and converts it into a pandas DataFrame.

    Parameters:
    parquet_file (str): The path to the Parquet file.

    Returns:
    pd.DataFrame: A DataFrame containing the data from the Parquet file.
    """
    parquet_dir = os.path.join(os.path.dirname(__file__), "..", "..", "parquet")
    parquet_file = os.path.join(parquet_dir, "data.parquet")

    try:
        data_frame = pd.read_parquet(parquet_file, engine="fastparquet")
        return data_frame
    except Exception as e:
        print(f"An error occurred while loading the Parquet file: {e}")
        return None
