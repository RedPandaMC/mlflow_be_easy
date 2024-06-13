"""Gets Data From Sklearn's Iris DataSet"""
from sklearn.datasets import load_iris
from pandas import DataFrame
from prefect import flow, task


def extract() -> DataFrame:
    """
    Fetches the iris dataset from sklearn.

    Returns:
        df (DataFrame): A pandas DataFrame containing the iris dataset.
    """
    df = load_iris(as_frame=True)
    return df