"""Gets Data From Sklearn's Iris DataSet"""
from sklearn.datasets import load_iris
import pandas as pd
from prefect import flow, task

def extract() -> pd.DataFrame:
    """
    Fetches the iris dataset from sklearn.

    Returns:
        df (pd.DataFrame): A pandas DataFrame containing the iris dataset.
    """
    X, y = load_iris(return_X_y=True)
    feature_names = load_iris().feature_names
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    return df
