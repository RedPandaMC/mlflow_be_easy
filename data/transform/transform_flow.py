"""Adds Synthetic Data To Sklearns Iris DataSet"""

from imblearn.over_sampling import SMOTE
from pandas import DataFrame, concat
from prefect import flow, task


def transform(data_frame: DataFrame) -> DataFrame:
    """
    Transforms a given dataset by oversampling the minority classes using SMOTE.

    Args:
        data_frame (DataFrame): Input dataframe with 'target' column.

    Returns:
        DataFrame: Transformed dataframe with synthetic data added.
    """
    synthetic_data_frame = concat(
        [data_frame, data_frame.sample(frac=0.5, replace=True)], ignore_index=True
    )

    X = synthetic_data_frame.drop("target", axis=1)
    y = synthetic_data_frame["target"]

    oversampler = SMOTE()
    X_oversampled, y_oversampled = oversampler.fit_resample(X, y)

    data_frame_oversampled = DataFrame(X_oversampled, columns=X.columns)

    columns_to_ratio = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ]

    # Calculate ratios for each pair of columns
    for i in range(len(columns_to_ratio)):
        for j in range(i + 1, len(columns_to_ratio)):
            col1 = columns_to_ratio[i]
            col2 = columns_to_ratio[j]
            ratio_col_name = f"{col1[:-5]}/{col2[:-5]}"  # New column name for the ratio
            data_frame_oversampled[ratio_col_name] = (
                data_frame_oversampled[col1] / data_frame_oversampled[col2]
            )

    data_frame_oversampled["target"] = y_oversampled

    data_frame_oversampled = data_frame_oversampled.sort_values(
        by="target", ascending=True
    ).reset_index(drop=True)
    return data_frame_oversampled
