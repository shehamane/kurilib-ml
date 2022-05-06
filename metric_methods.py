import math

import numpy as np

import pandas as pd
import metrics


class kNearestNeighbors:
    __metrics_map = {'euclidian': metrics.EuclidianNorm,
                     'manhattan': metrics.ManhattanNorm,
                     'cosine': metrics.CosineDistance}

    def __init__(self, k, X: pd.DataFrame, y: pd.Series, metric: str = 'euclidian',
                 search_method='exhaustive', choice_method='maxentry', window_width=None):

        if k > 0:
            self.__k = k
        else:
            raise Exception('K must be positive')

        if search_method in ['exhaustive', 'kdtree']:
            self.__search_method = search_method
        else:
            raise Exception(f'No such search method: {search_method}')

        self.__metrics = self.__class__.__metrics_map[metric]

        if choice_method in ['maxentry', 'weighted', 'kernel-regression']:
            self.__choice_method = choice_method
        else:
            raise Exception(f'No such choice method: {choice_method}')

        if window_width is not None and window_width > 0 or choice_method == 'maxentry':
            self.__window_width = window_width
        else:
            raise Exception('Window width must be positive')

        if X.shape[0] != y.shape[0]:
            raise Exception('X and y are incompatible shapes')
        if choice_method != 'kernel-regression':
            self.__classes = y.unique()
        if search_method == 'exhaustive':
            self.__data = X.to_numpy()
            self.__target = y.to_numpy()

    def set_metrics(self, metric):
        self.__metrics = self.__class__.__metrics_map[metric]

    def set_choice_method(self, choice_method):
        self.__choice_method = choice_method

    def __exhaustive_search(self, x: np.ndarray) -> (np.ndarray, np.ndarray):
        neighbors_idx = []
        neighbors_dists = []
        idx = 0
        for obj in self.__data:
            dist = self.__metrics.get_distance(x, obj)
            if len(neighbors_idx) >= self.__k:
                for i in range(self.__k):
                    if dist < neighbors_dists[i]:
                        neighbors_dists[i] = dist
                        neighbors_idx[i] = idx
                        break
            else:
                neighbors_dists.append(dist)
                neighbors_idx.append(idx)
            idx += 1
        return self.__data[neighbors_idx], self.__target[neighbors_idx]

    def __search(self, x: np.ndarray) -> (np.ndarray, np.ndarray):
        if self.__search_method == 'exhaustive':
            return self.__exhaustive_search(x)

    @staticmethod
    def __gaussian_kernel(x):
        return math.pow(math.e, -2 * math.pow(x, 2)) / math.sqrt(2 * math.pi)

    def __nadarai_watson(self, x, neighbors: np.ndarray, targets: np.ndarray):
        return np.sum(
            [self.__gaussian_kernel(
                self.__metrics.get_distance(x, neighbors[i]) / self.__window_width
            ) * targets[i] for i in range(0, neighbors.shape[0])]
        ) \
               / np.sum(
            [self.__gaussian_kernel(
                self.__metrics.get_distance(x, neighbor) / self.__window_width
            ) for neighbor in neighbors]
        )

    @staticmethod
    def __get_most_freq_class(targets: np.ndarray):
        unique, pos = np.unique(targets, return_inverse=True)
        maxpos = np.bincount(pos).argmax()
        return unique[maxpos]

    def __get_most_heavy_class(self, x, neighbors: np.ndarray, targets: np.ndarray):

        max_sum = 0
        max_class = None
        for cls in self.__classes:
            weighted_sum = 0
            for i, target in enumerate(targets):
                if target == cls:
                    weighted_sum += self.__gaussian_kernel(self.__metrics.get_distance(x, neighbors[i])
                                                           / self.__window_width)
            if weighted_sum > max_sum:
                max_sum = weighted_sum
                max_class = cls

        return max_class

    def __predict_once(self, x: np.ndarray):
        neighbors, targets = None, None
        if self.__search_method == 'exhaustive':
            neighbors, targets = self.__search(x)

        if self.__choice_method == 'maxentry':
            return self.__get_most_freq_class(targets)
        elif self.__choice_method == 'weighted':
            return self.__get_most_heavy_class(x, neighbors, targets)
        elif self.__choice_method == 'kernel-regression':
            return self.__nadarai_watson(x, neighbors, targets)

    def predict(self, X: pd.DataFrame):
        if self.__choice_method == 'kernel-regression':
            predictions = np.empty(shape=X.shape[0])
        else:
            predictions = np.empty(shape=X.shape[0], dtype='O')
        X = X.to_numpy()
        for i, obj in enumerate(X):
            predictions[i] = self.__predict_once(obj)
        return predictions
