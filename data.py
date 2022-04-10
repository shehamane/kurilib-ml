import numpy as np
import pandas as pd


def add_ones_feature(design_matrix: np.ndarray):
    return np.c_[design_matrix, np.ones(design_matrix.shape[0])]


def minimax(arr: np.ndarray):
    return (arr - arr.min()) / (arr.max() - arr.min())


def normalize_columns(df: pd.DataFrame, columns: [str], method='minimax'):
    for col_name in columns:
        df[col_name] = pd.Series(minimax(df[col_name]))
