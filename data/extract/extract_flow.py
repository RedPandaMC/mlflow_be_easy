"""Gets Data From Sklearn's Iris DataSet"""

import pandas as pd
from prefect import flow, task
from sklearn.datasets import load_iris


def extract() -> pd.DataFrame:
    """
    Fetches the iris dataset from sklearn.

    Returns:
        data_frame (pd.DataFrame): A pandas DataFrame containing the iris dataset.
    """
    X, y = load_iris(return_X_y=True)
    feature_names = load_iris().feature_names
    data_frame = pd.DataFrame(X, columns=feature_names)
    data_frame["target"] = y
    return data_frame
