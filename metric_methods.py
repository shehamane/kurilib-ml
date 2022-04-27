import numpy as np

import pandas as pd
import metrics


class kNearestNeighbors:
    __metrics_map = {'euclidian': metrics.EuclidianNorm}

    def __init__(self, k, X: pd.DataFrame, y: pd.Series, metric: str = 'euclidian',
                 search_method='exhaustive', choice_method='maxentry'):
        self.__k = k
        self.__search_method = search_method
        self.__metrics = self.__class__.__metrics_map[metric]
        self.__choice_method = choice_method
        if search_method == 'exhaustive':
            self.__data = X.to_numpy()
            self.__target = y.to_numpy()

    def set_metrics(self, metric):
        self.__metrics = self.__class__.metrics_map[metric]

    def set_choice_method(self, choice_method):
        self.__choice_method = choice_method

    def __exhaustive_search(self, x: np.ndarray):
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
        return neighbors_idx

    def __get_most_freq_target(self, objects_idx: list):
        targets = self.__target[objects_idx]
        unique, pos = np.unique(targets, return_inverse=True)
        maxpos = np.bincount(pos).argmax()
        return unique[maxpos]

    def __predict_once(self, x: np.ndarray):
        if self.__search_method == 'exhaustive':
            neighbors_idx = self.__exhaustive_search(x)

        if self.__choice_method == 'maxentry':
            return self.__get_most_freq_target(neighbors_idx)

    def predict(self, X: pd.DataFrame):
        predictions = np.empty(shape=X.shape[0], dtype='O')
        # X = X.to_numpy()
        for i, obj in enumerate(X):
            predictions[i] = self.__predict_once(obj)
        return predictions
