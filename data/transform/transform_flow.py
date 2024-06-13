"""Adds Synthetic Data To Sklearns Iris DataSet"""
from pandas import DataFrame, concat
from prefect import flow, task
from imblearn.over_sampling import SMOTE

def transform(df: DataFrame) -> DataFrame:
    """
    Transforms a given dataset by oversampling the minority classes using SMOTE.

    Args:
        df (DataFrame): Input dataframe with 'target' column.

    Returns:
        DataFrame: Transformed dataframe with synthetic data added.
    """
    synthetic_df = concat([df, df.sample(frac=0.5, replace=True)], ignore_index=True)

    X = synthetic_df.drop('target', axis=1)
    y = synthetic_df['target']

    oversampler = SMOTE()
    X_oversampled, y_oversampled = oversampler.fit_resample(X, y)

    df_oversampled = DataFrame(X_oversampled, columns=X.columns)
    
    columns_to_ratio = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

    # Calculate ratios for each pair of columns
    for i in range(len(columns_to_ratio)):
        for j in range(i + 1, len(columns_to_ratio)):
            col1 = columns_to_ratio[i]
            col2 = columns_to_ratio[j]
            ratio_col_name = f'{col1[:-5]}/{col2[:-5]}'  # New column name for the ratio
            df_oversampled[ratio_col_name] = df_oversampled[col1] / df_oversampled[col2]

    df_oversampled['target'] = y_oversampled

    df_oversampled = df_oversampled.sort_values(by='target', ascending=True).reset_index(drop=True)
    return df_oversampled