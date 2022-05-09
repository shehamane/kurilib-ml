import math

import numpy as np
from scipy.spatial import KDTree

import pandas as pd
import metrics


class kNearestNeighbors:
    __metrics_map = {'euclidian': metrics.EuclidianNorm,
                     'manhattan': metrics.ManhattanNorm,
                     'cosine': metrics.CosineDistance}

    def __init__(self, k, X: pd.DataFrame, y: pd.Series, metric: str = 'euclidian',
                 search_method='kdtree', choice_method='popular'):
        if k > 0:
            self.__k = k
        else:
            raise Exception('K must be positive')

        if search_method in ['exhaustive', 'kdtree']:
            self.__search_method = search_method
        else:
            raise Exception(f'No such search method: {search_method}')

        self.__metrics = self.__class__.__metrics_map[metric]

        if choice_method in ['popular', 'weighted', 'kernel-regression']:
            self.__choice_method = choice_method
        else:
            raise Exception(f'No such choice method: {choice_method}')

        if X.shape[0] != y.shape[0]:
            raise Exception('X and y are incompatible shapes')
        if choice_method != 'kernel-regression':
            self.__classes = y.unique()
        if search_method == 'kdtree':
            if metric == 'cosine':
                raise Exception('Cosine metric and k-d tree are incompatible')
            self.__kdtree = KDTree(X.to_numpy())

        if metric == 'euclidian':
            self.__norm = 2
        elif metric == 'manhattan':
            self.__norm = 1

        self.__X = X.to_numpy()
        self.__y = y.to_numpy()

    def set_metrics(self, metric):
        self.__metrics = self.__class__.__metrics_map[metric]

    def set_choice_method(self, choice_method):
        self.__choice_method = choice_method

    @staticmethod
    def __gaussian_kernel(x):
        return math.pow(math.e, -2 * math.pow(x, 2)) / math.sqrt(2 * math.pi)

    def __nadarai_watson(self, targets, distances, window_width):
        return np.sum(
            [self.__gaussian_kernel(distances[i] / window_width) * targets[i]
             for i in range(0, self.__k)]
        ) / np.sum(
            [self.__gaussian_kernel(distances[i] / window_width)
             for i in range(0, self.__k)]
        )

    @staticmethod
    def __get_most_popular_class(targets: np.ndarray):
        unique, pos = np.unique(targets, return_inverse=True)
        maxpos = np.bincount(pos).argmax()
        return unique[maxpos]

    def __get_most_heavy_class(self, targets: np.ndarray, distances, window_width):

        max_sum = 0
        max_class = None
        for cls in self.__classes:
            weighted_sum = 0
            for i, target in enumerate(targets):
                if target == cls:
                    weighted_sum += self.__gaussian_kernel(distances[i] / max(window_width, 10-6))
            if weighted_sum > max_sum:
                max_sum = weighted_sum
                max_class = cls

        return max_class

    def __choice_class(self, tt, dd):
        predicts = None
        if self.__choice_method == 'popular':
            predicts = [self.__get_most_popular_class(t) for t in tt]
        elif self.__choice_method == 'weighted':
            predicts = [self.__get_most_heavy_class(tt[i][:-1], dd[i][:-1], dd[i][-1])
                        for i in range(0, tt.shape[0])]
        elif self.__choice_method == 'kernel-regression':
            predicts = [self.__nadarai_watson(tt[i][:-1], dd[i][:-1], dd[i][-1])
                        for i in range(0, tt.shape[0])]
        return predicts

    def predict(self, X: pd.DataFrame):
        X = X.to_numpy()

        predicts = None
        if self.__search_method == 'kdtree':
            dd, ii = self.__kdtree.query(X, k=self.__k + int(self.__choice_method != 'popular'),
                                         p=self.__norm)
            tt = np.array([self.__y[i] for i in ii])
            predicts = self.__choice_class(tt, dd)

        return np.array(predicts)
