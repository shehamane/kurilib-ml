import copy
from abc import ABC

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from kurilib.base import Model
from kurilib.data import add_ones_feature, allocate_positive_class, pos_neg_allocate
from kurilib.quality_functional import MSE, LogisticLoss, sigmoid, MAE, loss_names_map


class AnalyticalSolution(Model):
    w: np.ndarray

    def __init__(self):
        pass

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = add_ones_feature(X)
        self.w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = add_ones_feature(X)
        return X.dot(self.w)


class GradientDescent(Model, ABC):
    def __init__(self, step: float, tolerance: float,
                 regularization: str, reg_lmb: float, descent_method: str, loss: str):
        self.w = None
        if descent_method != 'fastest':
            if step is not None and step > 0:
                self._step = step
            else:
                raise Exception('The descent step is not specified')
        if tolerance is None or tolerance > 0:
            self._tolerance = tolerance
        else:
            raise Exception('Tolerance must be positive')
        self.__reg_lmb = reg_lmb
        if regularization in ['L1', 'L2', None]:
            self.__regularization = regularization
        else:
            raise Exception('No such regularization method')
        if descent_method in ['const', 'normalization const', 'fastest']:
            if descent_method == 'fastest' and loss == LogisticLoss:
                raise Exception('Fastest descent and logistic loss are incompatible')
            self.__descent_method = descent_method
        else:
            raise Exception('No such descent method')
        if loss not in loss_names_map.keys():
            raise Exception('No such loss-function')
        else:
            self._loss = loss_names_map[loss]

    def _get_gradient(self, X, y):
        if self._loss == MSE:
            gradient = (2 / X.shape[0]) * X.T.dot(X.dot(self.w) - y)
        elif self._loss == LogisticLoss:
            gradient = -(X.T.dot(y - sigmoid(X.dot(self.w))))
        elif self._loss == MAE:
            judge = np.sign(X.dot(self.w) - y)
            gradient = np.sum(X * judge.reshape(X.shape[0], 1) / X.shape[0], axis=0)
        else:
            raise Exception('Gradient cannot be found for this loss-function')

        if 'normalization' in self.__descent_method:
            gradient /= np.linalg.norm(gradient)
        return gradient

    def _init_w(self, n):
        self.w = -1 / (2 * n) + np.random.rand(n) * 1 / n

    def _regularize(self, N: int):
        if self.__regularization == 'L2':
            self.w -= (2 * self.__reg_lmb / N) * self.w
        elif self.__regularization == 'L1':
            self.w -= (self.__reg_lmb / N) * np.sign(self.w)

    def _get_step(self, X, y, grad) -> float:
        if 'const' in self.__descent_method:
            return self._step
        elif self.__descent_method == 'fastest':
            if self._loss == LogisticLoss:
                return minimize(
                    lambda step: LogisticLoss.get_loss(X, y, self.w - step * grad),
                    x0=np.array([0]),
                    bounds=[[0, None]]
                ).x
            else:
                return minimize(
                    lambda step: self._loss.get_loss(X.dot(self.w - step * grad), y),
                    x0=np.array([0]),
                    bounds=[[0, None]]
                ).x

    def predict(self, X: pd.DataFrame):
        if type(X) is not np.ndarray:
            X = X.to_numpy()

        X = add_ones_feature(X)
        if self._loss == LogisticLoss:
            return sigmoid(X.dot(self.w))
        return X.dot(self.w)


class StandardGradientDescent(GradientDescent):
    def __init__(self, step, S: int, tolerance: float = 1, regularization=None, reg_lmb=None,
                 descent_method: str = 'const', loss: str = 'MSE'):
        super().__init__(step, tolerance, regularization, reg_lmb, descent_method, loss)
        if S > 0:
            self.__S = S
        else:
            raise Exception('S must be positive')

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if type(X) is not np.ndarray:
            X = X.to_numpy()
        if type(y) is not np.ndarray:
            y = y.to_numpy()
        X = add_ones_feature(X)

        self._init_w(X.shape[1])

        y_pred = X.dot(self.w)
        i = 0
        while i < self.__S and (self._loss == LogisticLoss or self._loss.get_loss(y_pred, y) > self._tolerance):
            grad = self._get_gradient(X, y)
            self.w -= self._get_step(X, y, grad) * grad
            self._regularize(X.shape[0])
            y_pred = X.dot(self.w)
            i += 1


class StochasticGradientDescent(GradientDescent):
    def __init__(self, step: float, eras: int, batch_size: int, tolerance: float = 1, loss: str = 'MSE',
                 descent_method: str = 'const', regularization: str = None, reg_lmb: float = None):
        super().__init__(step, tolerance, regularization, reg_lmb, descent_method, loss)
        if eras > 0:
            self.__eras = eras
        else:
            raise Exception('Eras number must be positive')
        if batch_size > 0:
            self.__batch_size = batch_size
        else:
            raise Exception('Batch size must be positive')

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if type(X) is not np.ndarray:
            X = X.to_numpy()
        if type(y) is not np.ndarray:
            y = y.to_numpy()
        X = add_ones_feature(X)

        self._init_w(X.shape[1])
        for era in range(0, self.__eras):
            i = self.__batch_size
            while i <= X.shape[0]:
                X_batch = X[i - self.__batch_size: i]
                y_batch = y[i - self.__batch_size: i]
                grad = self._get_gradient(X_batch, y_batch)
                self.w -= self._get_step(X_batch, y_batch, grad) * grad
                self._regularize(X_batch.shape[0])

                if MSE.get_loss(X_batch.dot(self.w), y_batch) < self._tolerance and self._loss != LogisticLoss:
                    return
                i += self.__batch_size

            X_batch = X[i - self.__batch_size: X.shape[0]]
            y_batch = y[i - self.__batch_size: X.shape[0]]
            grad = self._get_gradient(X_batch, y_batch)
            self.w -= self._get_step(X_batch, y_batch, grad) * grad
            self._regularize(X_batch.shape[0])

            if MSE.get_loss(X_batch.dot(self.w), y_batch) < self._tolerance:
                return


class OneVsAllClassifier(Model):
    def __init__(self, base_classifier):
        if type(base_classifier) not in [StandardGradientDescent, StochasticGradientDescent]:
            raise Exception('This classifier is not supported')
        self.__base_classifier = base_classifier
        self.__classifiers = None
        self.__classes = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if type(X) is not np.ndarray:
            X = X.to_numpy()

        if y.dtype != 'category':
            raise Exception('Target must be categorical')
        self.__classes = y.unique()
        self.__classifiers = [copy.deepcopy(self.__base_classifier) for _ in range(0, self.__classes.size)]
        for idx, cls in enumerate(self.__classes):
            target = allocate_positive_class(y, cls)
            self.__classifiers[idx].fit(X, target)

    def predict(self, X: pd.DataFrame):
        if type(X) is not np.ndarray:
            X = X.to_numpy()

        probas = np.zeros(shape=(self.__classes.size, X.shape[0]))
        for idx, classifier in enumerate(self.__classifiers):
            probas[idx] = classifier.predict(X)
        return np.array([self.__classes[idx] for idx in np.argmax(probas, axis=0)])


class AllVsAllClassifier(Model):
    def __init__(self, base_classifier):
        if type(base_classifier) not in [StandardGradientDescent, StochasticGradientDescent]:
            raise Exception('This classifier is not supported')
        self.__base_classifier = base_classifier
        self.__classifiers = None
        self.__classes = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if y.dtype != 'category':
            raise Exception('Target must be categorical')
        self.__classes = y.unique()
        self.__classifiers = []
        for i, lcls in enumerate(self.__classes[:-1]):
            for j, rcls in enumerate(self.__classes[i + 1:]):
                X_new, y_new = pos_neg_allocate(X, y, lcls, rcls)
                self.__classifiers.append(copy.deepcopy(self.__base_classifier))
                self.__classifiers[-1].fit(X_new.to_numpy(), y_new.to_numpy())

    def predict(self, X: pd.DataFrame):
        if type(X) is not np.ndarray:
            X = X.to_numpy()

        idx = 0
        predictions = np.empty(shape=(X.shape[0], len(self.__classifiers)), dtype='O')
        final_predictions = np.empty(shape=X.shape[0], dtype='O')
        for i, lcls in enumerate(self.__classes[:-1]):
            for rcls in self.__classes[i + 1:]:
                probas = self.__classifiers[idx].predict(X)
                predictions[:, idx] = np.array([lcls if proba > 0.5 else rcls for proba in probas])
                idx += 1

        for i, prediction in enumerate(predictions):
            unique, pos = np.unique(prediction, return_inverse=True)
            maxpos = np.bincount(pos).argmax()
            final_predictions[i] = unique[maxpos]

        return pd.Series(final_predictions, dtype='category')
