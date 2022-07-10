import numpy as np
import pandas as pd


def add_ones_feature(design_matrix: np.ndarray):
    return np.c_[design_matrix, np.ones(design_matrix.shape[0])]


def minimax(arr: np.ndarray):
    return (arr - arr.min()) / (arr.max() - arr.min())


def normalize_columns(df: pd.DataFrame, columns: [str]):
    for col_name in columns:
        df[col_name] = pd.Series(minimax(df[col_name]))


def allocate_positive_class(feature: pd.Series, positive: str):
    feature_np = feature.to_numpy()
    return pd.Series(np.where(feature_np == positive, 1, 0))


def pos_neg_allocate(X: pd.DataFrame, y: pd.Series, positive_class: str, negative_class: str) -> (
        np.ndarray, np.ndarray):
    positive_indexes = y[y == positive_class].index
    negative_indexes = y[y == negative_class].index
    X = pd.concat([X.loc[positive_indexes], X.loc[negative_indexes]], axis=0)
    y = pd.concat([y[positive_indexes], y[negative_indexes]])
    y = pd.Series(np.where(y == positive_class, 1, 0))
    return X, y