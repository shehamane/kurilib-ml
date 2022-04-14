import numpy as np
import pandas as pd


def add_ones_feature(design_matrix: np.ndarray):
    return np.c_[design_matrix, np.ones(design_matrix.shape[0])]


def minimax(arr: np.ndarray):
    return (arr - arr.min()) / (arr.max() - arr.min())


def normalize_columns(df: pd.DataFrame, columns: [str], method='minimax'):
    for col_name in columns:
        df[col_name] = pd.Series(minimax(df[col_name]))


def allocate_positive_class(feature: pd.Series, positive: str):
    feature_np = feature.to_numpy()
    return pd.Series(np.where(feature_np == positive, 1, 0))
