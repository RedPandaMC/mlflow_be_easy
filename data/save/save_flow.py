"""Saves Dataframe to Parquet File"""
from prefect import flow, task
import pandas as pd
import os

def save(df: pd.DataFrame) -> bool:
    """
    This function saves the given DataFrame to a Parquet file located in a
    directory two levels up from the current file's directory, under a 'parquet' folder.
    The Parquet file is named 'data.parquet'. If the directory does not exist,
    it will be created. The function uses the 'fastparquet' engine to write the file.

    Parameters:
    df (pd.DataFrame): The DataFrame to save.

    Returns:
        bool: True if the file was successfully saved, False otherwise.
    """
    success = False

    parquet_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'parquet')
    parquet_file = os.path.join(parquet_dir, 'data.parquet')
    
    os.makedirs(parquet_dir, exist_ok=True)
    
    df.to_parquet(parquet_file, index=False, engine='fastparquet')
    
    if os.path.isfile(parquet_file):
        success = True
    
    return success
